import torch
from torch.autograd import Variable
import time
import numpy as np
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy, average_gradients, average_tensor, pickup_max, vis_masks


def val_epoch(epoch, data_loader, model, criterion, args, logger):
    if args.local_rank == 0:
        print('validation at epoch {}'.format(epoch))

    batch_time = AverageMeter()
    model_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    un_losses1 = AverageMeter()
    un_losses2 = AverageMeter()
    un_losses3 = AverageMeter()
    top1_accuracies1 = AverageMeter()
    topk_accuracies1 = AverageMeter()
    if len(args.n_classes) == 2:
        top1_accuracies2 = AverageMeter()
        top1_accuracies3 = AverageMeter()
        topk_accuracies2 = AverageMeter()
        topk_accuracies3 = AverageMeter()
    if args.visualize:
        vis_step = np.random.randint(0, len(data_loader)-1, size=1)[0]

    model.eval()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        dist.barrier()
        inputs = inputs.cuda()
        if len(args.n_classes) == 1:
            targets = targets.cuda()
        if len(args.n_classes) == 2:
            verb_targets = targets[0].cuda()
            noun_targets = targets[1].cuda()
            
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        with torch.no_grad():
            logits, un_losses, masks = model(inputs)
        model_time.update(time.time() - end_time)
        
        if len(args.n_classes) == 1:
            cls_loss = criterion(logits, targets)
        elif len(args.n_classes) == 2:
            cls_loss = criterion(logits[0], verb_targets) + criterion(logits[1], noun_targets)
        loss = 0
        loss += cls_loss
        assert len(un_losses) == len(args.loss_weight)
        for j in range(len(un_losses)):
            if args.loss_weight[j] != 0:
                loss += args.loss_weight[j] * un_losses[j].mean()
            if j == 0:
                un_losses1.update(args.loss_vis_weight[j] * un_losses[j].mean().item(), inputs.size(0))
            elif j == 1:
                un_losses2.update(args.loss_vis_weight[j] * un_losses[j].mean().item(), inputs.size(0))
            elif j == 2:
                un_losses3.update(args.loss_vis_weight[j] * un_losses[j].mean().item(), inputs.size(0))
        if len(args.n_classes) == 1:
            _, acc = calculate_accuracy(logits, targets, args.top_k)
        elif len(args.n_classes) == 2:
            verb_correct, verb_acc = calculate_accuracy(logits[0], verb_targets, args.top_k)
            noun_correct, noun_acc = calculate_accuracy(logits[1], noun_targets, args.top_k)
            top_1_acc = (verb_correct[:,0]&noun_correct[:,0]).float().sum().item() / verb_targets.size(0)
            top_k_acc = (verb_correct.sum(-1)*noun_correct.sum(-1)).float().sum().item() / verb_targets.size(0)

        if args.visualize and i == vis_step and args.local_rank == 0:
            ind = np.random.randint(0, inputs.size(0), size=1)[0]
            vis_input = inputs[ind,:,:3,:,:] if args.modality == 'rgb+flow' else inputs[ind]
            vis_masks(vis_input.data.cpu().numpy(), masks[ind].data.cpu().numpy(), args.vis_path, 'val', epoch)
        
        cls_loss = average_tensor(cls_loss, args.world_size)
        cls_losses.update(cls_loss.item(), inputs.size(0) * args.world_size)
        loss = average_tensor(loss, args.world_size)
        losses.update(loss.item(), inputs.size(0) * args.world_size)
        if len(args.n_classes) == 1:
            acc[0] = average_tensor(acc[0], args.world_size)
            acc[1] = average_tensor(acc[1], args.world_size)
            top1_accuracies1.update(acc[0].item(), inputs.size(0) * args.world_size)
            topk_accuracies1.update(acc[1].item(), inputs.size(0) * args.world_size)
        elif len(args.n_classes) == 2:
            verb_acc[0] = average_tensor(verb_acc[0], args.world_size)
            verb_acc[1] = average_tensor(verb_acc[1], args.world_size)
            noun_acc[0] = average_tensor(noun_acc[0], args.world_size)
            noun_acc[1] = average_tensor(noun_acc[1], args.world_size)
            top_1_acc = average_tensor(top_1_acc, args.world_size)
            top_k_acc = average_tensor(top_k_acc, args.world_size)
            top1_accuracies1.update(verb_acc[0].item(), inputs.size(0) * args.world_size)
            topk_accuracies1.update(verb_acc[1].item(), inputs.size(0) * args.world_size)
            top1_accuracies2.update(noun_acc[0].item(), inputs.size(0) * args.world_size)
            topk_accuracies2.update(noun_acc[1].item(), inputs.size(0) * args.world_size)
            top1_accuracies3.update(top_1_acc.item(), inputs.size(0) * args.world_size)
            topk_accuracies3.update(top_k_acc.item(), inputs.size(0) * args.world_size)


        if (i + 1) % args.log_step == 0 and args.local_rank == 0:
            if len(args.n_classes) == 1:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                      'Model {model_time.val:.3f} ({model_time.avg:.3f}) | '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                      'CLSLoss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\n'
                      'UnLoss1 {un_loss1.val:.4f} ({un_loss1.avg:.4f}) | '
                      'UnLoss2 {un_loss2.val:.4f} ({un_loss2.avg:.4f}) | '
                      'UnLoss3 {un_loss3.val:.4f} ({un_loss3.avg:.4f})\n'
                      'Top1_Acc {top1_acc1.val:.4f} ({top1_acc1.avg:.4f}) | '
                      'Topk_Acc {topk_acc1.val:.4f} ({topk_acc1.avg:.4f})\n'.format(
                          epoch,
                          i + 1,
                          len(data_loader),
                          batch_time=batch_time,
                          model_time=model_time,
                          loss=losses,
                          cls_loss=cls_losses,
                          un_loss1=un_losses1,
                          un_loss2=un_losses2,
                          un_loss3=un_losses3,
                          top1_acc1=top1_accuracies1,
                          topk_acc1=topk_accuracies1), flush=True)
            elif len(args.n_classes) == 2:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                      'Model {model_time.val:.3f} ({model_time.avg:.3f}) | '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                      'CLSLoss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\n'
                      'UnLoss1 {un_loss1.val:.4f} ({un_loss1.avg:.4f}) | '
                      'UnLoss2 {un_loss2.val:.4f} ({un_loss2.avg:.4f}) | '
                      'UnLoss3 {un_loss3.val:.4f} ({un_loss3.avg:.4f})\n'
                      'Top1_Acc1 {top1_acc1.val:.4f} ({top1_acc1.avg:.4f}) | '
                      'Topk_Acc1 {topk_acc1.val:.4f} ({topk_acc1.avg:.4f}) | '
                      'Top1_Acc2 {top1_acc2.val:.4f} ({top1_acc2.avg:.4f}) | '
                      'Topk_Acc2 {topk_acc2.val:.4f} ({topk_acc2.avg:.4f}) | '
                      'Top1_Acc3 {top1_acc3.val:.4f} ({top1_acc3.avg:.4f}) | '
                      'Topk_Acc3 {topk_acc3.val:.4f} ({topk_acc3.avg:.4f})\n'.format(
                          epoch,
                          i + 1,
                          len(data_loader),
                          batch_time=batch_time,
                          model_time=model_time,
                          loss=losses,
                          cls_loss=cls_losses,
                          un_loss1=un_losses1,
                          un_loss2=un_losses2,
                          un_loss3=un_losses3,
                          top1_acc1=top1_accuracies1,
                          topk_acc1=topk_accuracies1,
                          top1_acc2=top1_accuracies2,
                          topk_acc2=topk_accuracies2,
                          top1_acc3=top1_accuracies3,
                          topk_acc3=topk_accuracies3), flush=True)

    if args.local_rank == 0:
        if len(args.n_classes) == 1:
            logger.log({
                    'epoch': epoch,
                    'loss': losses.avg,
                    'cls_loss': cls_losses.avg,
                    'un_loss1': un_losses1.avg,
                    'un_loss2': un_losses2.avg,
                    'un_loss3': un_losses3.avg,
                    'top1_acc1': top1_accuracies1.avg,
                    'topk_acc1': topk_accuracies1.avg
                })
        elif len(args.n_classes) == 2:
            logger.log({
                    'epoch': epoch,
                    'loss': losses.avg,
                    'cls_loss': cls_losses.avg,
                    'un_loss1': un_losses1.avg,
                    'un_loss2': un_losses2.avg,
                    'un_loss3': un_losses3.avg,
                    'top1_acc1': top1_accuracies1.avg,
                    'topk_acc1': topk_accuracies1.avg,
                    'top1_acc2': top1_accuracies2.avg,
                    'topk_acc2': topk_accuracies2.avg,
                    'top1_acc3': top1_accuracies3.avg,
                    'topk_acc3': topk_accuracies3.avg
                })

    dist.barrier()

    return losses.avg
