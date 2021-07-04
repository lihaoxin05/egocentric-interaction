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
    def path(self):
        return self._data[0]

    @property
    def start_frame(self):
        return int(self._data[1])
        
    @property
    def stop_frame(self):
        return int(self._data[2])

    @property
    def action_label(self):
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


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'frame_{:06d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            assert False

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video, local_rank):
    list_file = os.path.join(annotation_path, subset)
    data = [VideoRecord(x.strip().split(' ')) for x in open(list_file)]

    dataset = []
    for i in range(len(data)):
        if i % 2000 == 0 and local_rank == 0:
            print('dataset loading [{}/{}]'.format(i, len(data)))

        start_frame = data[i].start_frame
        stop_frame = data[i].stop_frame
        if stop_frame < start_frame:
            assert False, 'Number of frames cannot be negetive.'

        all_noun_label = [int(item) for item in data[i].all_noun_label_list[1:-1].strip().split(',')]
        
        sample = {
            'video': os.path.join(root_path, data[i].path),
            'video_id': data[i].path,
            'segment': [start_frame, stop_frame],
            'action': data[i].action_label,
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


class EGTEA_Gaze(data.Dataset):
    def __init__(self,
                 root_path, 
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader,
                 local_rank=0):
        self.data = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video, local_rank)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            new_clip = []
            for i in range(len(frame_indices)):
                img = clip[i]
                new_img = self.spatial_transform(img, 'rgb')
                new_clip.append(new_img)
            clip = new_clip
        clip = torch.stack(clip, 0)
        
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)
