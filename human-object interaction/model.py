import torch
from torch import nn
import torchvision
import torch.distributed as dist
# from models.i3res import I3ResNet
from models.base_net import base_net


def generate_model(args):
    assert args.model_depth in [18, 34, 50]

    if args.model_depth == 18:
        base_model = torchvision.models.resnet18(pretrained=True)
    elif args.model_depth == 34:
        base_model = torchvision.models.resnet34(pretrained=True)
    elif args.model_depth == 50:
        base_model = torchvision.models.resnet50(pretrained=True)
    
    model = base_net(args.modality, args.basenet_fixed_layers, args.n_classes, args.n_LSTM_layer, args.select_top_n, args.num_masks, args.mask_sigma, args.sample_duration)
    model.create_architecture(base_model)
    del base_model
    
    if args.pretrain_base_net_path:
        modality = ['rgb', 'flow']
        for i in range(2):
            md = modality[i]
            md_path = args.pretrain_base_net_path[i]
            if args.local_rank == 0:
                print('Loading pretrained base_net {}'.format(md_path))
            md_state_dict = torch.load(md_path, map_location={'cuda:0':'cuda:%d'%args.local_rank})['state_dict']
            local_state_dict = model.state_dict()
            count = 0
            for name, param in local_state_dict.items():
                if 'CNN_base_{}'.format(md) in name:
                    key = 'module.' + name.replace('CNN_base_{}'.format(md), 'CNN_base')
                    if key in md_state_dict:
                        input_param = md_state_dict[key].data
                        param.copy_(input_param)
                        count += 1
            if args.local_rank == 0:
                print('Loaded parameters: {}'.format(count))


    if args.pretrain_path:
        if args.local_rank == 0:
            print('Loading pretrained model {}'.format(args.pretrain_path))
        pretrain = torch.load(args.pretrain_path, map_location={'cuda:0':'cuda:%d'%args.local_rank})
        assert args.arch == pretrain['arch'], 'Unmatched model from pretrained path.'
        state_dict = pretrain['state_dict']
        local_state = model.state_dict()
        count = 0
        for name, param in local_state.items():
            load_pretrain = False
            for scope in args.pretrain_scope:
                if scope in name:
                    load_pretrain = True
                    break
            if load_pretrain:
                key = 'module.' + name
                if key in state_dict:
                    input_param = state_dict[key].data
                    param.copy_(input_param)
                    count += 1
                else:
                    assert False, "no pretrain {}".format(key)
        if args.local_rank == 0:
            print('Loaded parameters: {}'.format(count))
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    parameters = []
    arch_parameters = []
    for key, value in dict(model.named_parameters()).items():
        train = True
        if args.fix_scope:
            for scope in args.fix_scope:
                if scope in key:
                    train = False
                    break
        if train and value.requires_grad:
            if 'architect' not in key:
                parameters += [{'params':[value]}]
            else:
                arch_parameters += [{'params':[value]}]
    return model, parameters, arch_parameters

