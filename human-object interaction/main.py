import os
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import TemporalSampling
from target_transforms import ActionLabel, VerbLabel, NounLabel, AllNoun, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set, get_search_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test
import torch.distributed as dist
# import pdb


if __name__ == '__main__':
    args = parse_opts()
    ## for distributed dataloeader
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    args.world_size = torch.cuda.device_count()
    torch.cuda.set_device(args.local_rank)    
    dist.init_process_group(backend='nccl', init_method='env://')
    ## random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dist.barrier()

    args.arch = 'resnet-{}'.format(args.model_depth)
    args.mean = get_mean(args.modality)
    args.std = get_std(args.modality)
    if args.local_rank == 0:
        print(args, flush=True)
        with open(os.path.join(args.result_path, 'opts.json'), 'w') as opt_file:
            json.dump(vars(args), opt_file)

    model, parameters, arch_parameters = generate_model(args)
    if args.local_rank == 0:
        print(model, flush=True)
    criterion = nn.CrossEntropyLoss().cuda()

    if args.resume_path:
        if os.path.isfile(args.resume_path):
            print('Loading checkpoint {}'.format(args.resume_path))
            checkpoint = torch.load(args.resume_path, map_location={'cuda:0':'cuda:%d'%args.local_rank})
            assert args.arch == checkpoint['arch']
            begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            assert False, 'Not found {}'.format(args.resume_path)
    else:
        begin_epoch = 1

    norm_method = Normalize(args.mean, args.std)
    if not args.no_train:
        assert args.train_crop in ['random', 'corner', 'center']
        if args.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(args.scale, args.sample_size)
        elif args.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(args.scale, args.sample_size)
        elif args.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(args.scale, args.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            Scale(args.scale_size),
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(), norm_method
        ])
        temporal_transform = TemporalSampling(args.sample_duration, args.sample_step)
        if len(args.n_classes) == 1:
            target_transform = ActionLabel()
            if args.local_rank == 0:
                train_logger = Logger(
                    os.path.join(args.result_path, 'train.log'),
                    ['epoch', 'loss', 'cls_loss', 'un_loss1', 'un_loss2', 'un_loss3', 'top1_acc1', 'topk_acc1', 'lr'])
            else:
                train_logger = None
        elif len(args.n_classes) == 2:
            target_transform = TargetCompose([VerbLabel(), NounLabel()])
            if args.local_rank == 0:
                train_logger = Logger(
                    os.path.join(args.result_path, 'train.log'),
                    ['epoch', 'loss', 'cls_loss', 'un_loss1', 'un_loss2', 'un_loss3', 'top1_acc1', 'topk_acc1', 'top1_acc2', 'topk_acc2', 'top1_acc3', 'topk_acc3', 'lr'])
            else:
                train_logger = None
        training_data = get_training_set(args, spatial_transform, temporal_transform, target_transform)
        train_sampler = torch.utils.data.distributed.DistributedSampler(training_data, num_replicas=args.world_size, rank=args.local_rank)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.n_threads,
            pin_memory=True)

        if args.nesterov:
            dampening = 0
        else:
            dampening = args.dampening
        optimizer = optim.SGD(
            parameters,
            lr=args.learning_rate,
            momentum=args.momentum,
            dampening=dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=args.lr_patience)
        if args.search:
            search_data = get_search_set(args, spatial_transform, temporal_transform, target_transform)
            search_sampler = torch.utils.data.distributed.DistributedSampler(search_data, num_replicas=args.world_size, rank=args.local_rank)
            search_loader = torch.utils.data.DataLoader(
                search_data,
                sampler=search_sampler,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                pin_memory=True)
            arch_optimizer = optim.Adam(
                arch_parameters,
                lr=args.arch_learning_rate,
                weight_decay=args.arch_weight_decay)
        else:
            search_loader = None
            arch_optimizer = None
    
    if not args.no_val:
        spatial_transform = Compose([
            Scale(args.scale_size),
            CenterCrop(args.sample_size),
            ToTensor(), norm_method
        ])
        temporal_transform = TemporalSampling(args.sample_duration, args.sample_step)
        if len(args.n_classes) == 1:
            target_transform = ActionLabel()
            if args.local_rank == 0:
                val_logger = Logger(
                    os.path.join(args.result_path, 'val.log'), ['epoch', 'loss', 'cls_loss', 'un_loss1', 'un_loss2', 'un_loss3', 'top1_acc1', 'topk_acc1'])
            else:
                val_logger = None
        elif len(args.n_classes) == 2:
            target_transform = TargetCompose([VerbLabel(), NounLabel()])
            if args.local_rank == 0:
                val_logger = Logger(
                    os.path.join(args.result_path, 'val.log'), ['epoch', 'loss', 'cls_loss', 'un_loss1', 'un_loss2', 'un_loss3', 'top1_acc1', 'topk_acc1', 'top1_acc2', 'topk_acc2', 'top1_acc3', 'topk_acc3'])
            else:
                val_logger = None
        validation_data = get_validation_set(args, spatial_transform, temporal_transform, target_transform)
        val_sampler = torch.utils.data.distributed.DistributedSampler(validation_data, num_replicas=args.world_size, rank=args.local_rank, shuffle=False)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            sampler=val_sampler,
            batch_size=args.batch_size,
            num_workers=args.n_threads,
            pin_memory=True)

    if args.local_rank == 0:
        print('RUN')
    
    for i in range(begin_epoch, args.n_epochs + 1):
        if not args.no_train:
            train_sampler.set_epoch(i)
            train_epoch(i, train_loader, search_loader, model, criterion, optimizer, arch_optimizer, args, train_logger)
            dist.barrier()
        if not args.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, args, val_logger)
            dist.barrier()

        if not args.no_train and not args.no_val:
            scheduler.step(validation_loss)

    if args.test:
        spatial_transform = Compose([
            Scale(args.scale_size),
            CenterCrop(args.sample_size),
            ToTensor(), norm_method
        ])
        temporal_transform = TemporalSampling(args.sample_duration, args.sample_step)
        if len(args.n_classes) == 1:
            target_transform = TargetCompose([VideoID(), ActionLabel()])
        elif len(args.n_classes) == 2:
            target_transform = TargetCompose([VideoID(), VerbLabel(), NounLabel()])

        test_data = get_test_set(args, spatial_transform, temporal_transform, target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.n_threads,
            pin_memory=True)
        test.test(test_loader, model, criterion, args)
