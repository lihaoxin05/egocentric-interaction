import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video_path',
        default='',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--annotation_path',
        default='',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--result_path',
        default='',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pretrain_path', 
        default='', 
        type=str, 
        help='Pretrained model (.pth)')
    parser.add_argument(
        '--pretrain_scope', 
        default='CNN_base', 
        type=str,
        nargs='+',
        help='Pretrained parameter scopes')
    parser.add_argument(
        '--pretrain_path_2', 
        default='', 
        type=str, 
        help='Pretrained model (.pth)')
    parser.add_argument(
        '--pretrain_scope_2', 
        default='CNN_base', 
        type=str,
        nargs='+',
        help='Pretrained parameter scopes')
    parser.add_argument(
        '--fix_scope', 
        default='', 
        type=str,
        nargs='+',
        help='Fixed parameter scopes')
    parser.add_argument(
        '--pretrain_base_net_path',
        default='',
        type=str,
        nargs='+',
        help='Pretrained base net (rgb and flow)')
    parser.add_argument(
        '--dataset',
        default='',
        type=str,
        help='Used dataset')
    parser.add_argument(
        '--modality',
        default='rgb',
        type=str,
        help='Modality (rgb | flow | rgb+flow)')
    parser.add_argument(
        '--n_classes',
        default=[0],
        type=int,
        nargs='+',
        help='Number of classes')
    parser.add_argument(
        '--scale_size',
        default=256,
        type=float,
        help='Scale size')
    parser.add_argument(
        '--sample_size',
        default=[224,224],
        type=int,
        nargs='+',
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--sample_step',
        default=20,
        type=int,
        help='Temporal sampling step.')
    parser.add_argument(
        '--scale',
        default=0.9,
        type=float,
        help='Scale for multiscale cropping')
    parser.add_argument(
        '--train_crop',
        default='random',
        type=str,
        help=
        'Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)'
    )
    parser.add_argument(
        '--crop_position_in_test',
        default='c',
        type=str,
        help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=5,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--dampening', default=0, type=float, help='dampening of SGD')
    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--weight_decay', 
        default=1e-3, 
        type=float, 
        help='Weight Decay')
    parser.add_argument('--search', action='store_true', help='Search parameters')
    parser.set_defaults(search=False)
    parser.add_argument(
        '--search_list',
        default='',
        type=str,
        help='Used subset for arch search')
    parser.add_argument(
        '--arch_learning_rate',
        default=0.001,
        type=float,
        help='Initial architecture learning rate')
    parser.add_argument(
        '--arch_weight_decay', 
        default=1e-3, 
        type=float, 
        help='Architecture Weight Decay')
    parser.add_argument(
        '--select_top_n', default=2, type=int, help='Number of selected connections')
    parser.add_argument(
        '--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=50,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--top_k',
        default=5,
        type=int,
        help='Top k criterion.')
    parser.add_argument(
        '--n_val_samples',
        default=2,
        type=int,
        help='Number of validation samples for each video')
    parser.add_argument(
        '--n_test_samples',
        default=3,
        type=int,
        help='Number of test samples for each video')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--train_list',
        default='',
        type=str,
        help='Used subset in train')
    parser.add_argument(
        '--val_list',
        default='',
        type=str,
        help='Used subset in val')
    parser.add_argument(
        '--test_list',
        default='',
        type=str,
        help='Used subset in test')
    parser.add_argument(
        '--save_test_result',
        action='store_true',
        help='If true, save the test scores.')
    parser.set_defaults(save_test_result=False)
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=5,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--log_step',
        default=20,
        type=int,
        help='Log is printed at every this step.')
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (18 | 34 | 50 | 101 | 152)')
    parser.add_argument(
        '--basenet_fixed_layers',
        default=0,
        type=int,
        help='Fix number of layers of base net')
    parser.add_argument(
        '--num_masks',
        default=6,
        type=int,
        help='Number of masks')
    parser.add_argument(
        '--mask_sigma', 
        default=0.5, 
        type=float, 
        help='Sigma for masks generation')
    parser.add_argument(
        '--n_LSTM_layer', 
        default=1, 
        type=int, 
        help='Number of LSTM layer')
    parser.add_argument(
        '--loss_weight',
        default=[1],
        type=float,
        nargs='+',
        help='Loss weight')
    parser.add_argument('--visualize', action='store_true', help='Visualize boxes')
    parser.set_defaults(visualize=False)
    parser.add_argument(
        '--vis_path', 
        default='', 
        type=str, 
        help='Visualization path')
    parser.add_argument(
        '--loss_vis_weight',
        default=[1],
        type=float,
        nargs='+',
        help='Loss weight')
    parser.add_argument(
        '--seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--master_port', type=str, default='98765')
    args = parser.parse_args()

    return args
