import torch
from .interaction_net import interaction_net
import copy


class base_net(interaction_net):
    def __init__(self, modality, basenet_fixed_layers, n_classes, n_layer, select_top_n, num_masks, mask_sigma, sample_duration):
        self.basenet_fixed_layers = basenet_fixed_layers
        self.modality = modality
        
        interaction_net.__init__(self, modality, n_classes, n_layer, select_top_n, num_masks, mask_sigma, sample_duration)

    def _init_modules(self, base_model):
        # Build base_model
        if self.modality == 'rgb':
            self.CNN_base = torch.nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool, base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4)
        elif self.modality == 'flow':
            conv1 = torch.nn.Conv2d(10, base_model.conv1.out_channels, kernel_size=base_model.conv1.kernel_size, stride=base_model.conv1.stride, padding=base_model.conv1.padding, bias=False)
            conv1.weight = torch.nn.Parameter(torch.stack([base_model.conv1.weight.mean(1)]*10, dim=1))
            self.CNN_base = torch.nn.Sequential(conv1, base_model.bn1, base_model.relu, base_model.maxpool, base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4)
        elif self.modality == 'rgb+flow':
            self.CNN_base_rgb = torch.nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool, base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4)
            base_model_copy = copy.deepcopy(base_model)
            conv1 = torch.nn.Conv2d(10, base_model_copy.conv1.out_channels, kernel_size=base_model_copy.conv1.kernel_size, stride=base_model_copy.conv1.stride, padding=base_model_copy.conv1.padding, bias=False)
            conv1.weight = torch.nn.Parameter(torch.stack([base_model_copy.conv1.weight.mean(1)]*10, dim=1))
            self.CNN_base_flow = torch.nn.Sequential(conv1, base_model_copy.bn1, base_model_copy.relu, base_model_copy.maxpool, base_model_copy.layer1, base_model_copy.layer2, base_model_copy.layer3, base_model_copy.layer4)
            del base_model
            del base_model_copy

        # Fix blocks
        if self.modality == 'rgb+flow':
            if self.basenet_fixed_layers > 0:
                for p in self.CNN_base_rgb[0].parameters(): p.requires_grad=False
                for p in self.CNN_base_rgb[1].parameters(): p.requires_grad=False
                for p in self.CNN_base_flow[0].parameters(): p.requires_grad=False
                for p in self.CNN_base_flow[1].parameters(): p.requires_grad=False
            if self.basenet_fixed_layers >= 4:
                for p in self.CNN_base_rgb[7].parameters(): p.requires_grad=False
                for p in self.CNN_base_flow[7].parameters(): p.requires_grad=False
            if self.basenet_fixed_layers >= 3:
                for p in self.CNN_base_rgb[6].parameters(): p.requires_grad=False
                for p in self.CNN_base_flow[6].parameters(): p.requires_grad=False
            if self.basenet_fixed_layers >= 2:
                for p in self.CNN_base_rgb[6].parameters(): p.requires_grad=False
                for p in self.CNN_base_flow[6].parameters(): p.requires_grad=False
            if self.basenet_fixed_layers >= 1:
                for p in self.CNN_base_rgb[4].parameters(): p.requires_grad=False
                for p in self.CNN_base_flow[4].parameters(): p.requires_grad=False

        else:
            if self.basenet_fixed_layers > 0:
                for p in self.CNN_base[0].parameters(): p.requires_grad=False
                for p in self.CNN_base[1].parameters(): p.requires_grad=False
            if self.basenet_fixed_layers >= 4:
                for p in self.CNN_base[7].parameters(): p.requires_grad=False
            if self.basenet_fixed_layers >= 3:
                for p in self.CNN_base[6].parameters(): p.requires_grad=False
            if self.basenet_fixed_layers >= 2:
                for p in self.CNN_base[5].parameters(): p.requires_grad=False
            if self.basenet_fixed_layers >= 1:
                for p in self.CNN_base[4].parameters(): p.requires_grad=False

        # Fix bn
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1 or classname.find('bn') != -1:
                for p in m.parameters(): p.requires_grad=False

        if self.modality == 'rgb+flow':
            if self.basenet_fixed_layers > 0:
                self.CNN_base_rgb[0].apply(set_bn_fix)
                self.CNN_base_rgb[1].apply(set_bn_fix)
                self.CNN_base_flow[0].apply(set_bn_fix)
                self.CNN_base_flow[1].apply(set_bn_fix)
            if self.basenet_fixed_layers >= 4:
                self.CNN_base_rgb[7].apply(set_bn_fix)
                self.CNN_base_flow[7].apply(set_bn_fix)
            if self.basenet_fixed_layers >= 3:
                self.CNN_base_rgb[6].apply(set_bn_fix)
                self.CNN_base_flow[7].apply(set_bn_fix)
            if self.basenet_fixed_layers >= 2:
                self.CNN_base_rgb[5].apply(set_bn_fix)
                self.CNN_base_flow[7].apply(set_bn_fix)
            if self.basenet_fixed_layers >= 1:
                self.CNN_base_rgb[4].apply(set_bn_fix)
                self.CNN_base_flow[7].apply(set_bn_fix)
        else:
            if self.basenet_fixed_layers > 0:
                self.CNN_base[0].apply(set_bn_fix)
                self.CNN_base[1].apply(set_bn_fix)
            if self.basenet_fixed_layers >= 4:
                self.CNN_base[7].apply(set_bn_fix)
            if self.basenet_fixed_layers >= 3:
                self.CNN_base[6].apply(set_bn_fix)
            if self.basenet_fixed_layers >= 2:
                self.CNN_base[5].apply(set_bn_fix)
            if self.basenet_fixed_layers >= 1:
                self.CNN_base[4].apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        torch.nn.Module.train(self, mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1 or classname.find('bn') != -1:
                    m.eval()
            
            if self.modality == 'rgb+flow':
                self.CNN_base_rgb.train()
                self.CNN_base_flow.train()
                if self.basenet_fixed_layers >= 1:
                    self.CNN_base_rgb[0].eval()
                    self.CNN_base_rgb[0].apply(set_bn_eval)
                    self.CNN_base_rgb[1].eval()
                    self.CNN_base_rgb[1].apply(set_bn_eval)
                    self.CNN_base_rgb[2].eval()
                    self.CNN_base_rgb[3].eval()
                    self.CNN_base_rgb[4].eval()
                    self.CNN_base_rgb[4].apply(set_bn_eval)
                    self.CNN_base_flow[0].eval()
                    self.CNN_base_flow[0].apply(set_bn_eval)
                    self.CNN_base_flow[1].eval()
                    self.CNN_base_flow[1].apply(set_bn_eval)
                    self.CNN_base_flow[2].eval()
                    self.CNN_base_flow[3].eval()
                    self.CNN_base_flow[4].eval()
                    self.CNN_base_flow[4].apply(set_bn_eval)
                if self.basenet_fixed_layers >= 2:
                    self.CNN_base_rgb[5].eval()
                    self.CNN_base_rgb[5].apply(set_bn_eval)
                    self.CNN_base_flow[5].eval()
                    self.CNN_base_flow[5].apply(set_bn_eval)
                if self.basenet_fixed_layers >= 3:
                    self.CNN_base_rgb[6].eval()
                    self.CNN_base_rgb[6].apply(set_bn_eval)
                    self.CNN_base_flow[6].eval()
                    self.CNN_base_flow[6].apply(set_bn_eval)
                if self.basenet_fixed_layers >= 4:
                    self.CNN_base_rgb[7].eval()
                    self.CNN_base_rgb[7].apply(set_bn_eval)
                    self.CNN_base_flow[7].eval()
                    self.CNN_base_flow[7].apply(set_bn_eval)
            else:
                self.CNN_base.train()
                if self.basenet_fixed_layers >= 1:
                    self.CNN_base[0].eval()
                    self.CNN_base[0].apply(set_bn_eval)
                    self.CNN_base[1].eval()
                    self.CNN_base[1].apply(set_bn_eval)
                    self.CNN_base[2].eval()
                    self.CNN_base[3].eval()
                    self.CNN_base[4].eval()
                    self.CNN_base[4].apply(set_bn_eval)
                if self.basenet_fixed_layers >= 2:
                    self.CNN_base[5].eval()
                    self.CNN_base[5].apply(set_bn_eval)
                if self.basenet_fixed_layers >= 3:
                    self.CNN_base[6].eval()
                    self.CNN_base[6].apply(set_bn_eval)
                if self.basenet_fixed_layers >= 4:
                    self.CNN_base[7].eval()
                    self.CNN_base[7].apply(set_bn_eval)
