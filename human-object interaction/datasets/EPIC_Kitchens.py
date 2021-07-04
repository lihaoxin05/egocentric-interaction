import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import copy
import json
import numpy as np

from utils import load_value_file


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def id(self):
        return self._data[0]
    
    @property
    def path(self):
        return self._data[1]

    @property
    def start_frame(self):
        return int(self._data[2])
        
    @property
    def stop_frame(self):
        return int(self._data[3])

    @property
    def verb_label(self):
        return int(self._data[4])
        
    @property
    def noun_label(self):
        return int(self._data[5])
        
    @property
    def all_noun_label_list(self):
        return self._data[6]


def pil_loader(path, modality):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if modality == 'rgb':
                return img.convert('RGB')
            elif modality == 'flow':
                return img.convert('L')
            else:
                assert False, 'Do not support modality: {}!!'.format(modality)


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        return pil_loader(path, modality=modality)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, modality, image_loader):
    video = []
    if modality == 'rgb':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'frame_{:010d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                assert False, 'Do not exist: {}'.format(image_path)
    elif modality == 'flow':
        for i in frame_indices:
            clip = []
            for f in range(i-2, i+3):
                stack = []
                for c in ['u', 'v']:
                    image_path = os.path.join(video_dir_path, c, 'frame_{:010d}.jpg'.format(f))
                    if os.path.exists(image_path):
                        stack.append(image_loader(image_path, modality))
                    else:
                        assert False, 'Do not exist: {}'.format(image_path)
                clip.append(stack)
            video.append(clip)
    else:
        assert False, 'Do not support modality: {}!!'.format(modality)

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video, local_rank):
    list_file = os.path.join(annotation_path, subset)
    data = [VideoRecord(x.strip().split(' ')) for x in open(list_file)]

    dataset = []
    for i in range(len(data)):
        if i % 10000 == 0 and local_rank == 0:
            print('dataset loading [{}/{}]'.format(i, len(data)))

        start_frame = data[i].start_frame
        stop_frame = data[i].stop_frame
        if stop_frame < start_frame:
            assert False, 'Number of frames cannot be negetive.'

        all_noun_label = [int(item) for item in data[i].all_noun_label_list[1:-1].strip().split(',')]
        
        sample = {
            'rgb_path': os.path.join(root_path, 'rgb', data[i].path),
            'flow_path': os.path.join(root_path, 'flow', data[i].path),
            'segment': [start_frame, stop_frame],
            'video_id': data[i].id,
            'verb': data[i].verb_label,
            'noun': data[i].noun_label,
            'all_noun': all_noun_label
        }

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(start_frame, stop_frame + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                for j in range(n_samples_for_each_video):
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list(range(start_frame, stop_frame + 1))
                    dataset.append(sample_j)
            else:
                assert False

    return dataset


class EPIC_Kitchens(data.Dataset):
    def __init__(self,
                 root_path, 
                 annotation_path,
                 subset,
                 modality,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader,
                 local_rank=0):
        self.data = make_dataset(root_path, annotation_path, subset, n_samples_for_each_video, local_rank)
        self.modality = modality
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        frame_indices = self.data[index]['frame_indices']
        
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        if 'rgb' in self.modality:
            path = self.data[index]['rgb_path']
            rgb_video = self.loader(path, frame_indices, 'rgb')
        if 'flow' in self.modality:
            max_frame_ind = self.data[index]['segment'][1]
            frame_ind = [min(max(3, ind // 2), max_frame_ind // 2 - 1) for ind in frame_indices]
            path = self.data[index]['flow_path']
            flow_video = self.loader(path, frame_ind, 'flow')

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            if 'rgb' in self.modality:
                new_rgb_video = []
                for i in range(len(rgb_video)):
                    img = rgb_video[i]
                    new_img = self.spatial_transform(img, 'rgb')
                    new_rgb_video.append(new_img)
                rgb_video = torch.stack(new_rgb_video, 0)
            if 'flow' in self.modality:
                new_flow_video = []
                for i in range(len(flow_video)):
                    clip = flow_video[i]
                    new_clip = []
                    for j in range(len(clip)):
                        stack = clip[j]
                        new_stack = []
                        for k in range(len(stack)):
                            img = stack[k]
                            new_img = self.spatial_transform(img, 'flow')
                            new_stack.append(new_img)
                        new_clip.append(torch.stack(new_stack, 0))
                    new_flow_video.append(torch.stack(new_clip, 0))
                flow_video = torch.stack(new_flow_video, 0)
        if self.modality == 'rgb':
            video = rgb_video
        elif self.modality == 'flow':
            video_len, clip_len, stack_len, channel, height, weight = flow_video.shape
            video = flow_video.view(video_len, clip_len*stack_len*channel, height, weight).contiguous()
        elif self.modality == 'rgb+flow':
            video_len, clip_len, stack_len, channel, height, weight = flow_video.shape
            flow_video = flow_video.view(video_len, clip_len*stack_len*channel, height, weight).contiguous()
            video = torch.cat([rgb_video, flow_video], 1)
        
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return video, target

    def __len__(self):
        return len(self.data)
