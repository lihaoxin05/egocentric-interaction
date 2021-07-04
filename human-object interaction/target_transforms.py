import random
import math


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst

class Label(object):

    def __call__(self, target):
        return target['label']

class ActionLabel(object):

    def __call__(self, target):
        return target['action']

class VerbLabel(object):

    def __call__(self, target):
        return target['verb']
        
class NounLabel(object):

    def __call__(self, target):
        return target['noun']
        
class AllNoun(object):

    def __call__(self, target):
        return target['all_noun']

class VideoID(object):

    def __call__(self, target):
        return target['video_id']
