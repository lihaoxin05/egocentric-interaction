import random
import math
        

class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices, step=1):
        out = frame_indices[:self.size*step:step]

        while len(out) < self.size:
            out.append(out[-1])

        return out


class TemporalCenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices, step=1):

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2)*step)
        end_index = min(begin_index + self.size*step, len(frame_indices))

        out = frame_indices[begin_index:end_index:step]

        while len(out) < self.size:
            out.append(out[-1])

        return out


class TemporalRandomCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices, step=1):

        rand_end = max(0, len(frame_indices) - self.size*step - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size*step, len(frame_indices))

        out = frame_indices[begin_index:end_index:step]

        while len(out) < self.size:
            out.append(out[-1])

        return out
        
        
class TemporalSampling(object):

    def __init__(self, size, step=1):
        self.size = size
        self.step = step

    def __call__(self, frame_indices):
        
        if 1 + (self.size - 1) * self.step >=len(frame_indices):
            step = (len(frame_indices) - 1) / (self.size - 1)
            jittering = step // 2 - 1
            out = []
            for i in range(self.size):
                if i > 0 and i < self.size:
                    position = step * i + random.randint(min(-jittering,0),max(jittering,0))
                    position = min(max(1, position), len(frame_indices) - 2)
                    out.append(frame_indices[int(position)])
                else:
                    out.append(frame_indices[min(int(step*i), len(frame_indices) - 1)])
        else:
            rand_end = max(0, len(frame_indices) - (self.size-1)*self.step - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + (self.size-1)*self.step + 1, len(frame_indices))

            out = frame_indices[begin_index:end_index:self.step]

            while len(out) < self.size:
                out.append(out[-1])

        return out
