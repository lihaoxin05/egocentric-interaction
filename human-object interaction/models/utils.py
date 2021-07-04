import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def feat2position(feat):
    batch, time, num, h, w = feat.shape
    point_h = torch.stack([torch.arange(h).float() / h]*w, dim=1).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()
    point_w = torch.stack([torch.arange(w).float() / w]*h, dim=0).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()
    out_h = torch.sum(feat * point_h, dim=[3,4]) / torch.sum(feat, dim=[3,4])
    out_w = torch.sum(feat * point_w, dim=[3,4]) / torch.sum(feat, dim=[3,4])
    var_h = torch.sum(feat * (point_h - out_h.unsqueeze(-1).unsqueeze(-1)) ** 2.0, dim=[3,4]) / torch.sum(feat, dim=[3,4])
    var_w = torch.sum(feat * (point_w - out_w.unsqueeze(-1).unsqueeze(-1)) ** 2.0, dim=[3,4]) / torch.sum(feat, dim=[3,4])
    mean = torch.stack((out_h, out_w), dim=-1)
    var = (var_h + var_w) ** 2
    del point_h, point_w, out_h, out_w, var_h, var_w
    return mean, var
    
def position2map(position, feat_size, sigma=0.1):
    h, w = feat_size
    point_h = torch.stack([torch.arange(h).float() / h]*w, dim=1).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()
    point_w = torch.stack([torch.arange(w).float() / w]*h, dim=0).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()
    _keypoint_diff = (torch.stack((point_h, point_w), dim=-1) - position.unsqueeze(-2).unsqueeze(-2)) ** 2.0
    sigma = 2.0 * sigma**2.0
    feat = torch.exp(-(_keypoint_diff[:,:,:,:,:,0] + _keypoint_diff[:,:,:,:,:,1]) / sigma)
    del point_h, point_w, _keypoint_diff
    return feat
    
