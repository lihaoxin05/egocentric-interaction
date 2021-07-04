from __future__ import division
import os
import random
import tensorflow as tf
import numpy as np

class DataLoader(object):
    def __init__(self, 
                 dataset_dir=None, 
                 dataset_list_file=None,
                 batch_size=None, 
                 img_height=None, 
                 img_width=None, 
                 num_scales=None,
                 frames_per_video=None,
                 frame_step=None,
                 sample_per_video=None):
        self.frames_dir = os.path.join(dataset_dir, 'frames')
        self.parsing_dir = os.path.join(dataset_dir, 'parsing')
        self.dataset_list_file = dataset_list_file
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_scales = num_scales
        self.frames_per_video = frames_per_video
        self.frame_step = frame_step
        self.sample_per_video = sample_per_video
        self.file_list = self.format_file_list(self.dataset_list_file)
        self.steps_per_epoch = int(len(self.file_list['video_file_list']) // self.batch_size)

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.file_list
        video_paths_queue, start_index_queue, end_index_queue, labels_queue = tf.train.slice_input_producer(
            [tf.convert_to_tensor(file_list['video_file_list'], dtype=tf.string),
             tf.convert_to_tensor(file_list['start_index_list'], dtype=tf.int64),
             tf.convert_to_tensor(file_list['end_index_list'], dtype=tf.int64),
             tf.convert_to_tensor(file_list['label_list'], dtype=tf.int64)],
            seed=seed, 
            shuffle=True)

        # Load images
        video_path = video_paths_queue
        start_index = start_index_queue
        end_index = end_index_queue
        labels = labels_queue
        max_step = tf.floor_div(end_index-start_index, tf.convert_to_tensor(self.frames_per_video, tf.int64))
        step = tf.minimum(tf.minimum(tf.convert_to_tensor(self.frame_step, tf.int64), 7), max_step)
        max_step = end_index-tf.convert_to_tensor(self.frames_per_video-1, tf.int64)*step-2
        random_bias = tf.random_uniform([1], minval=0, maxval=max_step, dtype=tf.int64)[0]
        frame_random = tf.random_uniform([self.frames_per_video - 2], minval=-step+1, maxval=tf.maximum(step-1,-step+2), dtype=tf.int64)
        frame_random = tf.concat([tf.zeros([1], dtype=tf.int64), frame_random, tf.zeros([1], dtype=tf.int64)], axis=0)
        
        image_buffer = self.read_video(video_path, start_index, random_bias, end_index, self.frames_per_video, step, frame_random)
        image_seq = [tf.image.convert_image_dtype(tf.image.decode_jpeg(img, channels=3), dtype=tf.float32)
                        for img in image_buffer]
        image_seq = tf.stack(image_seq)
        image_seq.set_shape([self.frames_per_video, 180, 320, 3])
        image_seq = tf.image.resize_area(image_seq, [self.img_height, self.img_width])
        image_seq.set_shape([self.frames_per_video, self.img_height, self.img_width, 3])
        
        # Load parsing
        parsing_buffer = self.read_parsing(video_path, start_index, random_bias, end_index, self.frames_per_video, step, frame_random)
        parsing_seq = [tf.ceil(tf.image.convert_image_dtype(tf.image.decode_png(img, channels=1), dtype=tf.float32))
                        for img in parsing_buffer]
        parsing_seq = tf.stack(parsing_seq)
        parsing_seq.set_shape([self.frames_per_video, 180, 320, 1])
        parsing_seq = tf.image.resize_nearest_neighbor(parsing_seq, [self.img_height, self.img_width])
        parsing_seq.set_shape([self.frames_per_video, self.img_height, self.img_width, 1])

        # Form training batches
        image_seq, parsing_seq, labels, video_path = tf.train.batch([image_seq, parsing_seq, labels, video_path], batch_size=self.batch_size, num_threads=8, capacity=4*self.batch_size)

        # Data augmentation
        image_all, parsing_all = self.data_augmentation(image_seq, parsing_seq, self.img_height, self.img_width)
        
        tgt_image_seq = image_all[:, :-1, :, :, :]
        tgt_parsing_seq = parsing_all[:, :-1, :, :, :]
        src_image_seq = image_all[:, 1:, :, :, :]
        return tgt_image_seq, src_image_seq, tgt_parsing_seq, labels, video_path

    def load_inference_batch(self):
        """Load a batch of inference instances.
        """
        # Load the list of infernece files into queues
        file_list = self.file_list
        
        video_paths_queue, start_index_queue, end_index_queue, labels_queue = tf.train.slice_input_producer(
            [tf.convert_to_tensor(file_list['video_file_list'], dtype=tf.string),
             tf.convert_to_tensor(file_list['start_index_list'], dtype=tf.int64),
             tf.convert_to_tensor(file_list['end_index_list'], dtype=tf.int64),
             tf.convert_to_tensor(file_list['label_list'], dtype=tf.int64)],
            shuffle=False)

        # Load images
        video_path = video_paths_queue
        start_index = start_index_queue
        end_index = end_index_queue
        labels = labels_queue
        max_step = tf.floor_div(end_index-start_index, tf.convert_to_tensor(self.frames_per_video, tf.int64))
        step = tf.minimum(tf.minimum(tf.convert_to_tensor(self.frame_step, tf.int64), 7), max_step)
        max_step = end_index-tf.convert_to_tensor(self.frames_per_video-1, tf.int64)*step-2
        random_bias = tf.linspace(tf.cast(tf.constant(0), tf.float32), tf.cast(max_step, tf.float32), self.sample_per_video)
        random_bias = tf.cast(random_bias, tf.int64)
        
        all_image_buffer = []
        all_parsing_buffer = []
        for samples in range(self.sample_per_video):
            frame_random = tf.random_uniform([self.frames_per_video - 2], minval=-step+1, maxval=tf.maximum(step-1,-step+2), dtype=tf.int64)
            frame_random = tf.concat([tf.zeros([1], dtype=tf.int64), frame_random, tf.zeros([1], dtype=tf.int64)], axis=0)
            
            image_buffer = self.read_video(video_path, start_index, random_bias[samples], end_index, self.frames_per_video, step, frame_random)
            all_image_buffer.append(image_buffer)
            parsing_buffer = self.read_parsing(video_path, start_index, random_bias[samples], end_index, self.frames_per_video, step, frame_random)
            all_parsing_buffer.append(parsing_buffer)
        
        all_image_seq = []
        all_parsing_seq = []
        for image_buffer, parsing_buffer in zip(all_image_buffer, all_parsing_buffer):
            image_seq = [tf.image.convert_image_dtype(tf.image.decode_jpeg(img, channels=3), dtype=tf.float32)
                            for img in image_buffer]
            parsing_seq = [tf.ceil(tf.image.convert_image_dtype(tf.image.decode_png(img, channels=1), dtype=tf.float32))
                            for img in parsing_buffer]
            image_seq = tf.stack(image_seq)
            parsing_seq = tf.stack(parsing_seq)
            image_seq.set_shape([self.frames_per_video, 180, 320, 3])
            parsing_seq.set_shape([self.frames_per_video, 180, 320, 1])
            image_seq = tf.image.resize_area(image_seq, [self.img_height, self.img_width])
            parsing_seq = tf.image.resize_nearest_neighbor(parsing_seq, [self.img_height, self.img_width])
            image_seq.set_shape([self.frames_per_video, self.img_height, self.img_width, 3])
            parsing_seq.set_shape([self.frames_per_video, self.img_height, self.img_width, 1])
            all_image_seq.append(image_seq)
            all_parsing_seq.append(parsing_seq)
            
        image_seq = tf.stack(all_image_seq)
        parsing_seq = tf.stack(all_parsing_seq)
        
        # Form inference batches
        image_seq, parsing_seq, labels, video_path = tf.train.batch([image_seq, parsing_seq, labels, video_path], batch_size=self.batch_size, num_threads=4, capacity=2*self.batch_size)
        image_seq = tf.reshape(image_seq, [self.batch_size*self.sample_per_video, self.frames_per_video, self.img_height, self.img_width, 3])
        parsing_seq = tf.reshape(parsing_seq, [self.batch_size*self.sample_per_video, self.frames_per_video, self.img_height, self.img_width, 1])
        
        # Data augmentation
        image_all, parsing_all = self.data_augmentation(image_seq, parsing_seq, self.img_height, self.img_width, False)
        
        tgt_image_seq = image_all[:, :-1, :, :, :]
        tgt_parsing_seq = parsing_all[:, :-1, :, :, :]
        src_image_seq = image_all[:, 1:, :, :, :]
        return tgt_image_seq, src_image_seq, tgt_parsing_seq, labels, video_path

    def read_video(self, fpath, start_index, random_bias, end_index, num_frame, step, frame_random):
        with tf.variable_scope('read_image'):
            images = []
            for i in range(num_frame):
              impath = tf.string_join([tf.constant(self.frames_dir), tf.constant('/'),
                                       fpath, tf.constant('/'), tf.constant('frame'),
                                       tf.as_string(start_index + random_bias + i * step + frame_random[i], width=6, fill='0'), tf.constant('.jpg')])
              img_str = tf.read_file(impath)
              images.append(img_str)
        return images

    def read_parsing(self, fpath, start_index, random_bias, end_index, num_frame, step, frame_random):
        with tf.variable_scope('read_image'):
            images = []
            for i in range(num_frame):
              impath = tf.string_join([tf.constant(self.parsing_dir), tf.constant('/'),
                                       fpath, tf.constant('/'), tf.constant('frame'),
                                       tf.as_string(start_index + random_bias + i * step + frame_random[i], width=6, fill='0'), tf.constant('.png')])
              img_str = tf.read_file(impath)
              images.append(img_str)
        return images

    def data_augmentation(self, im, parsing, out_h, out_w, train_mode=True):
        # Random scaling
        def random_scaling(im, parsing):
            num_images, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.2)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            parsing = tf.image.resize_nearest_neighbor(parsing, [out_h, out_w])
            return im, parsing

        # Random cropping
        def random_cropping(im, parsing, out_h, out_w):
            num_images, in_h, in_w, _ = tf.unstack(tf.shape(im))
            # batch_size, time_length, in_h, in_w, channel = im.get_shape().as_list()
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h, out_w)
            parsing = tf.image.crop_to_bounding_box(parsing, offset_y, offset_x, out_h, out_w)
            return im, parsing
            
        # random_color
        def random_color(im, parsing):
            hue_delta = 0.4 * np.random.rand(1)[0] - 0.2
            saturation_factor = 0.45 * np.random.rand(1)[0] + 0.8
            contrast_factor = 0.45 * np.random.rand(1)[0] + 0.8
            hue = lambda x: tf.image.adjust_hue(x, hue_delta)
            saturation = lambda x: tf.image.adjust_saturation(x, saturation_factor)
            contrast = lambda x: tf.image.adjust_contrast(x, contrast_factor)
            im = tf.map_fn(hue, im)
            im = tf.map_fn(saturation, im)
            im = tf.map_fn(contrast, im)
            return im, parsing
            
        # random_flip
        def random_flip(im, parsing):
            ran = tf.random_uniform([1])[0]
            flip_im = lambda x: tf.image.flip_left_right(x)
            im = tf.cond(ran > tf.convert_to_tensor(0.5), lambda: tf.map_fn(flip_im, im), lambda: im)
            parsing = tf.cond(ran > tf.convert_to_tensor(0.5), lambda: tf.map_fn(flip_im, parsing), lambda: parsing)
            return im, parsing
            
        # mean_subtraction
        def mean_subtraction(im, parsing):
            ## Mean substracting to match the pretrain model
            ## [0,255]
            # im = im * 255
            # _R_MEAN = 123.68
            # _G_MEAN = 116.78
            # _B_MEAN = 103.94
            # mean_vals = [_R_MEAN, _G_MEAN, _B_MEAN]
            # num_channels = im.get_shape().as_list()[-1]
            # channels = tf.split(im, num_channels, -1)
            # for i in range(num_channels):
                # channels[i] -= mean_vals[i]
            # im = tf.concat(channels, -1)
            ## [0,1]
            im = im * 2
            mean_vals = 1
            im = im - mean_vals
            return im, parsing
        
        if train_mode:
          batch_size, time_length, in_h, in_w, channel = im.get_shape().as_list()
          im = tf.reshape(im, [batch_size * time_length, in_h, in_w, channel])
          parsing = tf.reshape(parsing, [batch_size * time_length, in_h, in_w, 1])
          im, parsing = random_scaling(im, parsing)
          im, parsing = random_cropping(im, parsing, out_h, out_w)
          im, parsing = random_color(im, parsing)
          im, parsing = random_flip(im, parsing)
          im, parsing = mean_subtraction(im, parsing)
          im = tf.reshape(im, [batch_size, time_length, out_h, out_w, channel])
          parsing = tf.reshape(parsing, [batch_size, time_length, out_h, out_w, 1])
        else:
          im, parsing = mean_subtraction(im, parsing)
        
        return im, parsing

    def format_file_list(self, list_file):
        with open(list_file, 'r') as f:
            videos = f.readlines()
        video_dir = [x.strip('\n').split(' ')[0] for x in videos]
        if len(videos[0].split(' '))>2:
            start_frames = [int(x.strip('\n').split(' ')[1]) for x in videos]
            end_frames = [int(x.strip('\n').split(' ')[2]) for x in videos]
        else:
            start_frames = [1] * len(videos)
            end_frames = [91] * len(videos)
        labels = [int(x.strip('\n').split(' ')[-1]) for x in videos]
        
        all_list = {}
        all_list['video_file_list'] = video_dir
        all_list['start_index_list'] = start_frames
        all_list['end_index_list'] = end_frames
        all_list['label_list'] = labels
        return all_list
