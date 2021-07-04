import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
import json
from utils import AverageMeter, calculate_accuracy


def calculate_video_results(video_id, output_buffer, label, test_results, criterion, args):
    if len(args.n_classes) == 1:
        video_outputs = torch.stack(output_buffer)
        average_scores = torch.mean(video_outputs, dim=0, keepdim=True)
        video_results = {}
        video_results['label'] = label.numpy().tolist()                  
        if label == -1:
            label = torch.randint_like(label, 0, args.n_classes)
        label = torch.unsqueeze(label, 0)
        loss = criterion(average_scores, label)
        _, acc = calculate_accuracy(average_scores, label, args.top_k)
        for class_ind in range(average_scores.size(1)):
            video_results["%d"%class_ind] = average_scores[0, class_ind].numpy().tolist()
        test_results["results"][video_id] = video_results
    elif len(args.n_classes) == 2:
        verb_outputs = torch.stack(output_buffer[0])
        average_verb_scores = torch.mean(verb_outputs, dim=0, keepdim=True)
        verb_label = label[0]
        if verb_label == -1:
            verb_label = torch.randint_like(verb_label, 0, args.n_classes[0])
        verb_label = torch.unsqueeze(verb_label, 0)
        verb_loss = criterion(average_verb_scores, verb_label)
        _, verb_acc = calculate_accuracy(average_verb_scores, verb_label, args.top_k)
        verb_results = {}
        for class_ind in range(average_verb_scores.size(1)):
            verb_results["%d"%class_ind] = average_verb_scores[0, class_ind].numpy().tolist()
        
        noun_outputs = torch.stack(output_buffer[1])
        average_noun_scores = torch.mean(noun_outputs, dim=0, keepdim=True)
        noun_label = label[1]
        if noun_label == -1:
            noun_label = torch.randint_like(noun_label, 0, args.n_classes[1])
        noun_label = torch.unsqueeze(noun_label, 0)
        noun_loss = criterion(average_noun_scores, noun_label)
        _, noun_acc = calculate_accuracy(average_noun_scores, noun_label, args.top_k)
        noun_results = {}
        for class_ind in range(average_noun_scores.size(1)):
            noun_results["%d"%class_ind] = average_noun_scores[0, class_ind].numpy().tolist()
        
        video_results = {"verb": verb_results, "noun": noun_results}
        test_results["results"][video_id] = video_results
        loss = [verb_loss, noun_loss]
        acc = [verb_acc, noun_acc, [verb_acc[0]*noun_acc[0], verb_acc[1]*noun_acc[1]]]

    return loss, acc


def test(data_loader, model, criterion, args):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    model_time = AverageMeter()
    cls_losses1 = AverageMeter()
    top1_accuracies1 = AverageMeter()
    topk_accuracies1 = AverageMeter()
    if len(args.n_classes) == 2:
        cls_losses2 = AverageMeter()
        top1_accuracies2 = AverageMeter()
        top1_accuracies3 = AverageMeter()
        topk_accuracies2 = AverageMeter()
        topk_accuracies3 = AverageMeter()

    if len(args.n_classes) == 1:
        output_buffer = []
    elif len(args.n_classes) == 2:
        verb_buffer = []
        noun_buffer = []
    previous_video_id = ''
    previous_target = ''
    test_results = {"version": "0.1", "challenge": "action_recognition", "results": {}}
    
    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        inputs = Variable(inputs)
        
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        with torch.no_grad():
            logits, _, _ = model(inputs)
        model_time.update(time.time() - end_time)
        
        video_id = targets[0]
        if len(args.n_classes) == 1:
            video_targets = targets[1]
            logits = logits.cpu()
            batch_size = logits.size(0)
        elif len(args.n_classes) == 2:
            verb_targets = targets[1]
            noun_targets = targets[2]
            verb_logits = logits[0].cpu()
            noun_logits = logits[1].cpu()
            batch_size = verb_logits.size(0)

        for j in range(batch_size):
            if (not (i == 0 and j == 0) and video_id[j] != previous_video_id) or (i == len(data_loader)-1 and j == batch_size-1):
                if len(args.n_classes) == 2:
                    output_buffer = [verb_buffer, noun_buffer]
                loss, acc = calculate_video_results(previous_video_id, output_buffer, previous_target, test_results, criterion, args) 
                if len(args.n_classes) == 1:
                    output_buffer = []
                    cls_losses1.update(loss)
                    top1_accuracies1.update(acc[0])
                    topk_accuracies1.update(acc[1])
                elif len(args.n_classes) == 2:
                    verb_buffer = []
                    noun_buffer = []
                    cls_losses1.update(loss[0])
                    cls_losses2.update(loss[1])
                    top1_accuracies1.update(acc[0][0])
                    topk_accuracies1.update(acc[0][1])
                    top1_accuracies2.update(acc[1][0])
                    topk_accuracies2.update(acc[1][1])
                    top1_accuracies3.update(acc[2][0])
                    topk_accuracies3.update(acc[2][1])
            elif not (i == 0 and j == 0):
                if len(args.n_classes) == 1:
                    assert video_targets[j] == previous_target
                elif len(args.n_classes) == 2:
                    assert verb_targets[j] == previous_target[0]
                    assert noun_targets[j] == previous_target[1]
            previous_video_id = video_id[j]
            if len(args.n_classes) == 1:
                previous_target = video_targets[j]
                output_buffer.append(logits[j])
            elif len(args.n_classes) == 2:
                previous_target = [verb_targets[j], noun_targets[j]]
                verb_buffer.append(verb_logits[j])
                noun_buffer.append(noun_logits[j])

        if (i + 1) % args.log_step == 0:
            print('[{}]'.format(i + 1), flush=True)
            print('[{time.val:.4f}\{time.avg:.4f}]'.format(time=model_time), flush=True)
    if len(args.n_classes) == 1:
        print('Loss {loss.avg:.4f}\tTop1_Acc {top1_acc.avg:.4f}\tTopk_Acc {topk_acc.avg:.4f}\n'.format(loss=cls_losses1, top1_acc=top1_accuracies1, topk_acc=topk_accuracies1), flush=True)
    elif len(args.n_classes) == 2:
        print('Loss1 {loss1.avg:.4f}\tLoss2 {loss2.avg:.4f}\nTop1_Acc1 {top1_acc1.avg:.4f}\tTopk_Acc1 {topk_acc1.avg:.4f}\tTop1_Acc2 {top1_acc2.avg:.4f}\tTopk_Acc2 {topk_acc2.avg:.4f}\tTop1_Acc3 {top1_acc3.avg:.4f}\tTopk_Acc3 {topk_acc3.avg:.4f}\n'.format(loss1=cls_losses1, loss2=cls_losses2, top1_acc1=top1_accuracies1, topk_acc1=topk_accuracies1, top1_acc2=top1_accuracies2, topk_acc2=topk_accuracies2, top1_acc3=top1_accuracies3, topk_acc3=topk_accuracies3), flush=True)
    if args.save_test_result:
        with open(os.path.join(args.result_path, '{}.json'.format(args.test_list)), 'w') as f:
            json.dump(test_results, f)