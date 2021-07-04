import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import feat2position, position2map
import pdb

class interaction_net(nn.Module):
    def __init__(self, modality, n_classes, n_layer, select_top_n, num_mask, mask_sigma, sample_duration):
        super(interaction_net, self).__init__()
        self.modality = modality
        final_dim = 1024 if modality=='rgb+flow' else 512
        self.num_mask = num_mask
        self.mask_sigma = mask_sigma
        self.hourglass_net = hourglass_net(self.num_mask)
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        # self.down_dim = nn.Linear(final_dim, 256)
        self.rnn = relational_LSTM(512, 512, n_layer, 6, select_top_n, num_mask, 0.2)
        self.n_fc = len(n_classes)
        if self.n_fc == 1:
            self.n_classes = n_classes[0]
            self.fc = nn.Linear(512*3,self.n_classes)
        elif self.n_fc == 2:
            self.n_verb = n_classes[0]
            self.n_noun = n_classes[1]
            self.fc_verb = nn.Linear(512*3, self.n_verb)
            self.fc_noun = nn.Linear(512*3, self.n_noun)
        else:
            assert False, "Not support more than two classifiers."
        
    def forward(self, video, training=False, searching=False):
        batch_size = video.size(0)
        time_length = video.size(1)
        height = video.size(3)
        width = video.size(4)
        video = video.data #shape: [b,T,C,H,W]
        video = video.view(batch_size*time_length,-1,height,width).contiguous() #shape: [bT,C,H,W]
        loss = []
        ## feed video data to base model to obtain base feature map
        if self.modality == 'rgb+flow':
            rgb_video = video[:,:3,:,:]
            flow_video = video[:,3:,:,:]
            rgb_base_feat = self.CNN_base_rgb(rgb_video) #shape: [bT,c,h,w]
            flow_base_feat = self.CNN_base_flow(flow_video) #shape: [bT,c,h,w]
            base_feat = torch.cat([rgb_base_feat, flow_base_feat], 1) #shape: [bT,2c,h,w]
        else:
            base_feat = self.CNN_base(video) #shape: [bT,c,h,w]
        ## global and local feat
        global_feat = self.pooling(base_feat).squeeze().view(batch_size,time_length,-1).contiguous() #shape: [b,t,c]
        if self.modality == 'rgb+flow':
            weighting_map = self.hourglass_net(video[:,:3,:,:], rgb_base_feat.detach()).view(batch_size,time_length,-1,height,width).contiguous() #shape: [b,T,N,H,W]
        else:
            weighting_map = self.hourglass_net(video, base_feat.detach()).view(batch_size,time_length,-1,height,width).contiguous() #shape: [b,T,N,H,W]
        position_mean, position_var = feat2position(weighting_map) #shape: [b,T,N,2]
        cen_loss = position_var.mean()
        loss.append(cen_loss)
        pairwise_diff = torch.norm(torch.stack([position_mean]*self.num_mask, dim=2) - torch.stack([position_mean]*self.num_mask, dim=3), dim=-1)
        sep_loss = ((1 - torch.eye(self.num_mask).cuda()).unsqueeze(0).unsqueeze(0) * torch.exp(-pairwise_diff)).sum(-1).sum(-1).mean()
        loss.append(sep_loss)
        masks = position2map(position_mean, [base_feat.shape[2], base_feat.shape[3]], self.mask_sigma) #shape: [b,T,N,h,w]
        normalize_masks = masks / (torch.sum(masks, dim=[3,4], keepdim=True) + 1e-8)
        local_feat = (base_feat.view(batch_size,time_length,-1,base_feat.shape[2], base_feat.shape[3]).unsqueeze(2) * normalize_masks.unsqueeze(3)).sum(-1).sum(-1) #shape: [b,T,N,c]
        sem_loss = torch.norm(local_feat[:,:-1,:,:] - local_feat[:,1:,:,:], dim=-1).mean()
        loss.append(sem_loss)
        # global_feat = F.relu(self.down_dim(global_feat))
        # local_feat = F.relu(self.down_dim(local_feat))
        local_feat = self.rnn(local_feat, global_feat, training=training, searching=searching)[:,-1,:].contiguous() #shape: [b,c]
        # total_feat = torch.cat([global_feat.mean(1), local_feat], dim=-1)
        ## classification
        if self.n_fc == 1:
            logits = self.fc(local_feat)
        elif self.n_fc == 2:
            logits_verb = self.fc_verb(local_feat)
            logits_noun = self.fc_noun(local_feat)
            logits = [logits_verb, logits_noun]
        return logits, loss, masks

    def _init_weights(self):
        self.hourglass_net._init_weights()
        self.rnn._init_weights()
        if self.n_fc == 1:
            nn.init.kaiming_normal_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0)
        elif self.n_fc == 2:
            nn.init.kaiming_normal_(self.fc_verb.weight)
            nn.init.constant_(self.fc_verb.bias, 0)
            nn.init.kaiming_normal_(self.fc_noun.weight)
            nn.init.constant_(self.fc_noun.bias, 0)

    def create_architecture(self, base_model):
        self._init_modules(base_model)
        self._init_weights()


class hourglass_net(nn.Module):
    def __init__(self, output_channel):
        super(hourglass_net, self).__init__()
        self.activate = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.softmax = nn.Softmax(dim=1)
        
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv6 = nn.Conv2d(512, 128, 1)
        
        self.upconv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.upconv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.upconv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.upconv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.upconv1 = nn.Conv2d(8, output_channel, 3, padding=1)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, fmap):
        x = self.activate(self.conv1(input))
        x = self.pool(x)
        y_2 = x
        
        x = self.activate(self.conv2(x))
        x = self.pool(x)
        y_4 = x
        
        x = self.activate(self.conv3(x))
        x = self.pool(x)
        y_8 = x
        
        x = self.activate(self.conv4(x))
        x = self.pool(x)
        y_16 = x
        
        x = self.activate(self.conv5(x))
        x = self.pool(x)

        fmap = self.activate(self.conv6(fmap))
        x = x + fmap
        
        x = self.upsample(x)
        x = self.activate(self.upconv5(x))
        x = x + y_16
        
        x = self.upsample(x)
        x = self.activate(self.upconv4(x))
        x = x + y_8
        
        x = self.upsample(x)
        x = self.activate(self.upconv3(x))
        x = x + y_4

        x = self.upsample(x)
        x = self.activate(self.upconv2(x))
        x = x + y_2
        
        x = self.upsample(x)
        x = self.upconv1(x)
        x = self.softmax(x)
        return x


class vanilla_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, dropout):
        super(vanilla_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.dropout = dropout
        # lstm
        self.lstm = nn.LSTM(input_size, hidden_size, n_layer, batch_first=True, dropout=dropout)

    def forward(self, feat):
        batch_size = feat.shape[0]
        # lstm
        h0 = torch.zeros(self.n_layer, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(self.n_layer, batch_size, self.hidden_size).cuda()
        self.lstm.flatten_parameters()
        feat, _ = self.lstm(feat, (h0, c0))
        return feat


class dual_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, dropout):
        super(dual_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.dropout = dropout
        # separate groups
        self.separation_param = nn.Linear(input_size, 1)
        # lstm
        self.lstm_0 = nn.LSTM(input_size, hidden_size, n_layer, batch_first=True, dropout=dropout)
        self.lstm_1 = nn.LSTM(input_size, hidden_size, n_layer, batch_first=True, dropout=dropout)
        
    def _init_weights(self):
        nn.init.kaiming_normal_(self.separation_param.weight)
        nn.init.constant_(self.separation_param.bias, 0)
            
    def forward(self, feat, training):
        batch_size, time_length, num_mask, input_size = feat.shape
        # separation
        s_weight = F.softmax(self.separation_param(feat), 2) - 1 / num_mask
        if training:
            s_weight = F.sigmoid(10*s_weight)
        else:
            s_weight = torch.where(s_weight > torch.zeros_like(s_weight), torch.ones_like(s_weight), torch.zeros_like(s_weight))
        separation_weight = torch.cat([s_weight, 1-s_weight], dim=-1).contiguous()
        group_0 = (separation_weight[:,:,:,0].unsqueeze(-1) * feat).sum(2) / (separation_weight[:,:,:,0].unsqueeze(-1).sum(2) + 1e-8)
        group_1 = (separation_weight[:,:,:,1].unsqueeze(-1) * feat).sum(2) / (separation_weight[:,:,:,1].unsqueeze(-1).sum(2) + 1e-8)
        # lstm
        g0_h0 = torch.zeros(self.n_layer, batch_size, self.hidden_size).cuda()
        g0_c0 = torch.zeros(self.n_layer, batch_size, self.hidden_size).cuda()
        g1_h0 = torch.zeros(self.n_layer, batch_size, self.hidden_size).cuda()
        g1_c0 = torch.zeros(self.n_layer, batch_size, self.hidden_size).cuda()
        self.lstm_0.flatten_parameters()
        group_0n, _ = self.lstm_0(group_0.contiguous(), (g0_h0, g0_c0))
        self.lstm_1.flatten_parameters()
        group_1n, _ = self.lstm_1(group_1.contiguous(), (g1_h0, g1_c0))
        concat_group = torch.cat([group_0n, group_1n], dim=-1)
        return concat_group


class interactive_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, dropout):
        super(interactive_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.dropout = dropout
        # separate groups
        self.separation_param = nn.Linear(input_size, 1)
        # lstm weights
        for layer in range(n_layer):
            layer_input_size = input_size if layer == 0 else hidden_size
            params = [nn.Linear(layer_input_size, 4 * hidden_size)]
            param_names = ['lstm_param_x_t_{}']
            params.append(nn.Linear(hidden_size, 4 * hidden_size))
            param_names.append('lstm_param_h_t-1_{}')
            params.append(nn.Linear(hidden_size, 4 * hidden_size))
            param_names.append('lstm_param_hh_t-1_{}')
            param_names = [x.format(layer) for x in param_names]
            for name, param in zip(param_names, params):
                setattr(self, name, param)  # self.name = param
                
    def _init_weights(self):
        nn.init.kaiming_normal_(self.separation_param.weight)
        nn.init.constant_(self.separation_param.bias, 0)
        for i in range(self.n_layer):
            op = getattr(self, 'lstm_param_x_t_{}'.format(i))
            nn.init.kaiming_normal_(op.weight)
            nn.init.constant_(op.bias, 0)
            op = getattr(self, 'lstm_param_h_t-1_{}'.format(i))
            nn.init.kaiming_normal_(op.weight)
            nn.init.constant_(op.bias, 0)
            op = getattr(self, 'lstm_param_hh_t-1_{}'.format(i))
            nn.init.kaiming_normal_(op.weight)
            nn.init.constant_(op.bias, 0)
            
            
    def layer_forward(self, seq_feat, layer_num):
        batch_size, time_length, n_group, channel = seq_feat.shape
        hidden_states = torch.zeros(batch_size, 2, n_group, self.hidden_size).cuda()
        cell_state = torch.zeros(batch_size, n_group, self.hidden_size).cuda()
        store_hidden = []
        store_cell = []
        for i in range(time_length):
            hidden_states[:,0,:,:] = hidden_states[:,1,:,:]
            for j in range(n_group):
                candidate_state = []
                candidate_state.append(getattr(self, 'lstm_param_x_t_{}'.format(layer_num))(seq_feat[:,i,j,:].contiguous()))
                candidate_state.append(getattr(self, 'lstm_param_h_t-1_{}'.format(layer_num))(hidden_states[:,-2,j,:].contiguous()))
                candidate_state.append(getattr(self, 'lstm_param_hh_t-1_{}'.format(layer_num))(hidden_states[:,-2,(j+1)%2,:].contiguous()))
                candidate_state = torch.stack(candidate_state, -1).sum(-1)
                input_gate, output_gate, forget_gate, cell_candidate = torch.chunk(candidate_state, 4, dim=-1)
                cell_state[:,j,:] = F.sigmoid(forget_gate) * cell_state[:,j,:].clone() + F.sigmoid(input_gate) * F.tanh(cell_candidate)
                hidden_states[:,-1,j,:] = F.sigmoid(output_gate) * F.tanh(cell_state[:,j,:].clone())
            store_hidden.append(hidden_states[:,-1,:,:])
            store_cell.append(cell_state)
        store_hidden = torch.stack(store_hidden, 1)
        store_cell = torch.stack(store_cell, 1)
        return store_hidden, store_cell
                
            
    def forward(self, feat, training):
        batch_size, time_length, num_mask, input_size = feat.shape
        # separation        
        s_weight = F.softmax(self.separation_param(feat), 2) - 1 / num_mask
        if training:
            s_weight = F.sigmoid(10*s_weight)
        else:
            s_weight = torch.where(s_weight > torch.zeros_like(s_weight), torch.ones_like(s_weight), torch.zeros_like(s_weight))
        separation_weight = torch.cat([s_weight, 1-s_weight], dim=-1).contiguous()
        group_0 = (separation_weight[:,:,:,0].unsqueeze(-1) * feat).sum(2) / (separation_weight[:,:,:,0].unsqueeze(-1).sum(2) + 1e-8)
        group_1 = (separation_weight[:,:,:,1].unsqueeze(-1) * feat).sum(2) / (separation_weight[:,:,:,1].unsqueeze(-1).sum(2) + 1e-8)
        concat_group = torch.stack([group_0, group_1], dim=2) # shape: [b,T,2,C]
        # lstm
        for i in range(self.n_layer):
            if i == 0:
                hidden_states = concat_group
            else:
                hidden_states = F.dropout(hidden_states, p=self.dropout, training=training)
            hidden_states, cell_states = self.layer_forward(hidden_states, i) # shape: [b,T,2,h]
        return_feat = hidden_states.view(batch_size, time_length, -1).contiguous()
        return return_feat


class relational_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, n_candidate_connect, select_top_n, n_mask, dropout):
        super(relational_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_candidate_connect = n_candidate_connect
        self.select_top_n = select_top_n
        self.dropout = dropout
        # share feat
        self.local2share = nn.Linear(n_mask*input_size, hidden_size)
        # separate groups
        self.separation_param = nn.Linear(input_size, 1)
        # rnn weights
        for layer in range(n_layer):
            # candidate connection
            arch_params = nn.Linear(hidden_size, n_candidate_connect)
            arch_param_names = 'architect_param_{}'.format(layer)
            setattr(self, arch_param_names, arch_params)  # self.name = param
            layer_input_size = input_size if layer == 0 else hidden_size
            for j in range(2):
                params = [nn.Linear(layer_input_size, 5 * hidden_size)]
                param_names = ['lstm_param_x_t_%d_{}'%j]
                params.append(nn.Linear(hidden_size, 5 * hidden_size))
                param_names.append('lstm_param_h_t-1_%d_{}'%j)
                for connect in range(n_candidate_connect):
                    hidden_input_size = hidden_size if connect == 0 or connect == 2 or connect == 5 else layer_input_size
                    params.append(nn.Linear(hidden_input_size, 5 * hidden_size))
                    param_names.append('lstm_param_can_%d_%d_{}'%(connect, j))
                param_names = [x.format(layer) for x in param_names]
                for name, param in zip(param_names, params):
                    setattr(self, name, param)  # self.name = param
                
    def _init_weights(self):
        nn.init.kaiming_normal_(self.local2share.weight)
        nn.init.constant_(self.local2share.bias, 0)
        nn.init.kaiming_normal_(self.separation_param.weight)
        nn.init.constant_(self.separation_param.bias, 0)
        for i in range(self.n_layer):
            op = getattr(self, 'architect_param_{}'.format(i))
            nn.init.kaiming_normal_(op.weight, 0)
            nn.init.constant_(op.bias, 0)
            for j in range(2):
                op = getattr(self, 'lstm_param_x_t_{}_{}'.format(j,i))
                nn.init.kaiming_normal_(op.weight)
                nn.init.constant_(op.bias, 0)
                op = getattr(self, 'lstm_param_h_t-1_{}_{}'.format(j,i))
                nn.init.kaiming_normal_(op.weight)
                nn.init.constant_(op.bias, 0)
                for k in range(self.n_candidate_connect):
                    op = getattr(self, 'lstm_param_can_{}_{}_{}'.format(k,j,i))
                    nn.init.kaiming_normal_(op.weight)
                    nn.init.constant_(op.bias, 0)
            
    def layer_forward(self, seq_feat, share_feat, scene_feat, layer_num, searching=False):
        batch_size, time_length, n_group, channel = seq_feat.shape
        hidden_states = torch.zeros(batch_size, 3, n_group, self.hidden_size).cuda()
        cell_state = torch.zeros(batch_size, n_group, self.hidden_size).cuda()
        share_state = torch.zeros(batch_size, self.hidden_size).cuda()
        store_hidden = []
        store_cell = []
        store_share = []
        cum_obj_gate = []
        for i in range(time_length):
            hidden_states[:,0,:,:] = hidden_states[:,1,:,:]
            hidden_states[:,1,:,:] = hidden_states[:,2,:,:]
            weight = getattr(self, 'architect_param_{}'.format(layer_num))(hidden_states[:,-2,:,:].clone())
            if not searching:
                idx = torch.topk(weight, k=self.select_top_n, dim=-1)[1]
                weight = torch.zeros_like(weight).scatter_(-1, idx, 1/self.select_top_n)
            else:
                weight = F.softmax(weight, dim=-1)
            share_feat_gate = []
            for j in range(n_group):
                candidate_state = []
                candidate_state.append(getattr(self, 'lstm_param_x_t_{}_{}'.format(j,layer_num))(seq_feat[:,i,j,:].contiguous()))
                candidate_state.append(getattr(self, 'lstm_param_h_t-1_{}_{}'.format(j,layer_num))(hidden_states[:,-2,j,:].contiguous()))
                for k in range(self.n_candidate_connect):
                    op = getattr(self, 'lstm_param_can_{}_{}_{}'.format(k,j,layer_num))
                    if k == 0:
                        input = hidden_states[:,-3,j,:].contiguous()
                    elif k == 1:
                        input = seq_feat[:,i+1,j,:].contiguous() if i < time_length - 1 else torch.zeros_like(seq_feat[:,i,j,:].contiguous()).cuda()
                    elif k == 2:
                        input = hidden_states[:,-2,(j+1)%2,:].contiguous()
                    elif k == 3:
                        input = seq_feat[:,i,(j+1)%2,:].contiguous()
                    elif k == 4:
                        input = seq_feat[:,i+1,(j+1)%2,:].contiguous() if i < time_length - 1 else torch.zeros_like(seq_feat[:,i,(j+1)%2,:].contiguous()).cuda()
                    elif k == 5:
                        input = scene_feat[:,i,:].contiguous()
                    else:
                        assert False
                    candidate_state.append(weight[:,j,k].unsqueeze(-1) * op(input))
                candidate_state = torch.stack(candidate_state, -1).sum(-1)
                input_gate, forget_gate, output_gate, share_gate, cell_candidate = torch.chunk(candidate_state, 5, dim=-1)
                share_feat_gate.append(share_gate)
                cell_state[:,j,:] = (1 - F.sigmoid(forget_gate)) * cell_state[:,j,:].clone() + F.sigmoid(input_gate) * F.tanh(cell_candidate)
                hidden_states[:,-1,j,:] = F.sigmoid(output_gate) * F.tanh(cell_state[:,j,:].clone())
            share_feat_gate = F.sigmoid(torch.stack(share_feat_gate, -1).sum(-1))
            share_state = share_state + share_feat_gate * share_feat[:,i,:].clone()
            cum_obj_gate.append(share_feat_gate)
            store_share.append(share_state)
            store_hidden.append(hidden_states[:,-1,:,:])
            store_cell.append(cell_state)
        store_hidden = torch.stack(store_hidden, 1)
        store_cell = torch.stack(store_cell, 1)
        cum_obj_gate = torch.cumsum(torch.stack(cum_obj_gate, 1),1)
        store_share = torch.stack(store_share, 1) / cum_obj_gate
        return store_hidden, store_cell, store_share
                
            
    def forward(self, feat, context_feat, training=False, searching=False):
        batch_size, time_length, num_mask, input_size = feat.shape
        # share feature
        share_feat = F.relu(self.local2share(feat.view(batch_size, time_length, num_mask*input_size)))
        # separation
        s_weight = F.softmax(self.separation_param(feat), 2) - 1 / num_mask
        if training:
            s_weight = F.sigmoid(10*s_weight)
        else:
            s_weight = torch.where(s_weight > torch.zeros_like(s_weight), torch.ones_like(s_weight), torch.zeros_like(s_weight))
        separation_weight = torch.cat([s_weight, 1-s_weight], dim=-1).contiguous()
        group_0 = (separation_weight[:,:,:,0].unsqueeze(-1) * feat).sum(2) / (separation_weight[:,:,:,0].unsqueeze(-1).sum(2) + 1e-8) # shape: [b,T,C]
        group_1 = (separation_weight[:,:,:,1].unsqueeze(-1) * feat).sum(2) / (separation_weight[:,:,:,1].unsqueeze(-1).sum(2) + 1e-8)
        concat_group = torch.stack([group_0, group_1], dim=2) # shape: [b,T,2,C]
        # lstm
        retrun_hidden = []
        return_share = []
        for i in range(self.n_layer):
            if i == 0:
                hidden_states = concat_group
            else:
                hidden_states = F.dropout(hidden_states, p=self.dropout if searching else 0.0)
            hidden_states, _, share_feat = self.layer_forward(hidden_states.clone(), share_feat.clone(), context_feat, i, searching=searching) # shape: [b,T,2,h]
            retrun_hidden.append(hidden_states.view(batch_size, time_length, -1).contiguous())
            return_share.append(share_feat)
        return_feat = torch.cat([torch.stack(retrun_hidden, 0).mean(0), torch.stack(return_share, 0).mean(0)], dim=-1)
        return return_feat