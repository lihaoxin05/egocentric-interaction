import os
import csv
import pdb
import time
import skimage
import skimage.io
import skimage.transform
import skimage.color
import numpy as np
import torch
import torch.distributed as dist
from mean import get_mean, get_std


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(round(values[col], 7))

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets, topk=5):
    batch_size = targets.size(0)

    _, pred = outputs.topk(topk, 1, True)
    correct = pred.eq(targets.view(-1, 1))
    top_1_correct_elems = correct[:,0].float().sum().item()
    top_5_correct_elems = correct.float().sum().item()

    return correct, [top_1_correct_elems / batch_size, top_5_correct_elems / batch_size]
        
        
def average_gradients(model, world_size):
    size = float(world_size)
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

def average_tensor(loss, world_size):
    if not isinstance(loss, torch.Tensor):
        loss = torch.Tensor([loss]).cuda()
    dist.barrier()
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss /= world_size
    return loss

def pickup_max(t):
    if not isinstance(t, torch.Tensor):
        t = torch.Tensor([t]).cuda()
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t


def vis_masks(video, masks, path, mode, epoch):
    mean = np.array(get_mean('rgb')['rgb'])
    std = np.array(get_std('rgb')['rgb'])
    curr_time = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, mode)
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, '{:02d}-{}'.format(epoch, curr_time))
    if not os.path.isdir(path):
        os.mkdir(path)
    
    video = np.transpose(video, (0,2,3,1))
    height = video.shape[1]
    width = video.shape[2]
    masks = np.clip(np.sum(masks, axis=1), 0, 1)
    masks = np.transpose(skimage.transform.resize(np.transpose(np.squeeze(masks), (1,2,0)), (height, width)), (2,0,1))
    
    for i in range(video.shape[0]-1):
        frame = np.clip((video[i] * std + mean)*255, 0, 255).astype('uint8')
        frame_path = os.path.join(path, '%03d.jpg'%(i))
        skimage.io.imsave(frame_path, frame)
        
        mask = masks[i]
        mask_max = np.max(mask)
        mask_min = np.min(mask)
        mask = (mask - mask_min) / (mask_max - mask_min)
        mask = np.uint8(mask * 255)
        frame_path = os.path.join(path, '%03d_mask_max%.3f_min%.3f.jpg'%(i, mask_max, mask_min))
        skimage.io.imsave(frame_path, mask)
        