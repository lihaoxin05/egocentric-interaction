from datasets.EPIC_Kitchens import EPIC_Kitchens
from datasets.EGTEA_Gaze import EGTEA_Gaze
from datasets.SomethingSomething import STHV1


def get_training_set(args, spatial_transform, temporal_transform, target_transform):
    assert args.dataset in ['EPIC_Kitchens', 'EGTEA_Gaze', 'something_something_v1']

    if args.dataset == 'EPIC_Kitchens':
        training_data = EPIC_Kitchens(
            args.video_path,
            args.annotation_path,
            args.train_list,
            args.modality,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)
    elif args.dataset == 'EGTEA_Gaze':
        training_data = EGTEA_Gaze(
            args.video_path,
            args.annotation_path,
            args.train_list,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)
    elif args.dataset == 'something_something_v1':
        training_data = STHV1(
            args.video_path,
            args.annotation_path,
            args.train_list,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)
            
    return training_data

def get_validation_set(args, spatial_transform, temporal_transform, target_transform):
    assert args.dataset in ['EPIC_Kitchens', 'EGTEA_Gaze', 'something_something_v1']

    if args.dataset == 'EPIC_Kitchens':
        validation_data = EPIC_Kitchens(
            args.video_path,
            args.annotation_path,
            args.val_list,
            args.modality,
            n_samples_for_each_video=args.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)
    elif args.dataset == 'EGTEA_Gaze':
        validation_data = EGTEA_Gaze(
            args.video_path,
            args.annotation_path,
            args.val_list,
            n_samples_for_each_video=args.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)
    elif args.dataset == 'something_something_v1':
        validation_data = STHV1(
            args.video_path,
            args.annotation_path,
            args.val_list,
            n_samples_for_each_video=args.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)
    
    return validation_data


def get_test_set(args, spatial_transform, temporal_transform, target_transform):
    assert args.dataset in ['EPIC_Kitchens', 'EGTEA_Gaze', 'something_something_v1']

    if args.dataset == 'EPIC_Kitchens':
        test_data = EPIC_Kitchens(
            args.video_path,
            args.annotation_path,
            args.test_list,
            args.modality,
            n_samples_for_each_video=args.n_test_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)
    elif args.dataset == 'EGTEA_Gaze':
        test_data = EGTEA_Gaze(
            args.video_path,
            args.annotation_path,
            args.test_list,
            n_samples_for_each_video=args.n_test_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)
    elif args.dataset == 'something_something_v1':
        test_data = STHV1(
            args.video_path,
            args.annotation_path,
            args.test_list,
            n_samples_for_each_video=args.n_test_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)

    return test_data


def get_search_set(args, spatial_transform, temporal_transform, target_transform):
    assert args.dataset in ['EPIC_Kitchens', 'EGTEA_Gaze', 'something_something_v1']

    if args.dataset == 'EPIC_Kitchens':
        search_data = EPIC_Kitchens(
            args.video_path,
            args.annotation_path,
            args.search_list,
            args.modality,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)
    elif args.dataset == 'EGTEA_Gaze':
        search_data = EGTEA_Gaze(
            args.video_path,
            args.annotation_path,
            args.search_list,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)
    elif args.dataset == 'something_something_v1':
        search_data = STHV1(
            args.video_path,
            args.annotation_path,
            args.search_list,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            local_rank=args.local_rank)
            
    return search_data
