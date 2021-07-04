from __future__ import division
import tensorflow as tf
import time
import random
import numpy as np
from learner import Learner
import os

flags = tf.app.flags
flags.DEFINE_string("gpu_list", "0", "Comma sep list of gpus.")
################################### dataset flags ###################################
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("dataset_list_file", None, "Dataset list")
################################### training flags ###################################
flags.DEFINE_string("checkpoint_dir", None, "Directory name to save the checkpoints")
flags.DEFINE_string("checkpoint_file", None, "Specific checkpoint file to inference")
flags.DEFINE_string("visual_dir", "./inference_vis/", "Directory name to visulize")
flags.DEFINE_boolean("vis_flag", False, "Visulization or not.")
flags.DEFINE_integer("vis_freq", 10, "visualization every log_freq iterations")
flags.DEFINE_float("cls_loss_weight", 1e0, "Weight for classification error loss")
flags.DEFINE_float("parsing_loss_weight", 1e-1, "Weight for mask loss")
flags.DEFINE_float("pixel_loss_weight", 1e-1, "Weight for pixel error loss")
flags.DEFINE_float("smooth_loss_weight", 1e-1, "Weight for smoothness")
flags.DEFINE_float("base_net_weight_reg", 1e-5, "Weight for base_net parameters regularization")
flags.DEFINE_float("attention_net_weight_reg", 1e-4, "Weight for attention_net parameters regularization")
flags.DEFINE_float("motion_net_weight_reg", 1e-4, "Weight for motion_net parameters regularization")
flags.DEFINE_float("interaction_net_weight_reg", 1e-4, "Weight for interaction_net parameters regularization")
flags.DEFINE_float("keep_prob", 1.0, "Keep_prob for dropout.")
flags.DEFINE_integer("batch_size", 1, "The size of a batch")
flags.DEFINE_integer("sample_per_video", 4, "The numbers of samples per video")
flags.DEFINE_string("to_obs_loss", "cls_loss", "Which loss to observe.")
################################### model flags ###################################
flags.DEFINE_integer("num_scales", 3, "The number of scale")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 256, "Image width")
flags.DEFINE_integer("frames_per_video", 20, "frames_per_video")
flags.DEFINE_integer("frame_step", 5, "frame_step")
flags.DEFINE_integer("NEIGHBORHOOD_SIZE", 11, "neighborhood_size for correlation module")
flags.DEFINE_integer("STRIDE", 1, "stride for correlation module")
flags.DEFINE_integer("stack_feat_size", 128, "stack_size of non-correlation feature for correlation module")
flags.DEFINE_integer("hidden_size", 256, "Hidden size of LSTM")
flags.DEFINE_integer("num_class", 8, "Number of class.")
FLAGS = flags.FLAGS

def main(_):
    # seed = int(time.time())
    seed = 0
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(FLAGS.visual_dir):
        os.makedirs(FLAGS.visual_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    model = Learner()
    model.inference(FLAGS)

if __name__ == '__main__':
    tf.app.run()
