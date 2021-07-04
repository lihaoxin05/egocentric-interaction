def get_mean(modality):
    mean = {}
    if 'rgb' in modality:
        mean['rgb'] = [0.485, 0.456, 0.406]
    if 'flow' in modality:
        mean['flow'] = [0.5]
    return mean

def get_std(modality):
    std = {}
    if 'rgb' in modality:
        std['rgb'] = [0.229, 0.224, 0.225]
    if 'flow' in modality:
        std['flow'] = [0.226]
    return std
