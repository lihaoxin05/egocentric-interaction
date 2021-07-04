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
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


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
        image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print(image_path)
            assert False

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video):
    list_file = os.path.join(annotation_path, subset)
    data = [VideoRecord(x.strip().split(' ')) for x in open(list_file)]

    dataset = []
    for i in range(len(data)):
        if i % 10000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(data)))

        num_frames = data[i].num_frames
        if num_frames <= 0:
            assert False, 'Number of frames cannot be negetive.'
        
        sample = {
            'video': os.path.join(root_path, data[i].path),
            'video_id': data[i].path,
            'segment': [1, num_frames],
            'action': data[i].label,
        }

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, num_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                for j in range(n_samples_for_each_video):
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list(range(1, num_frames + 1))
                    dataset.append(sample_j)
            else:
                assert False

    return dataset


class STHV1(data.Dataset):
    def __init__(self,
                 root_path, 
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader):
        self.data = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video)

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
                new_img = self.spatial_transform(img)
                new_clip.append(new_img)
            clip = new_clip
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)
