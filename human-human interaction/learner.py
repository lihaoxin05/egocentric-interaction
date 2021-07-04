from __future__ import division
import os
import time
import math
import tensorflow as tf
import numpy as np
from skimage import io
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from nets import *
from utils import *

class Learner(object):
    def __init__(self):
        pass


    def build_dataloader(self):
        """Construct data loader."""
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.dataset_list_file,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_scales,
                            opt.frames_per_video,
                            opt.frame_step,
                            opt.sample_per_video)
        self.steps_per_epoch = math.ceil(loader.steps_per_epoch // self.gpu_num)
        return loader


    def data_preprocessing(self, tgt_image_seq, src_image_seq, parsing_seq):
        """Preprocessing images data."""
        batch_size, time_length, in_h, in_w, in_channel = tgt_image_seq.get_shape().as_list()
        tgt_image_all = tf.reshape(tgt_image_seq, [batch_size * time_length, in_h, in_w, in_channel])
        src_image_all = tf.reshape(src_image_seq, [batch_size * time_length, in_h, in_w, in_channel])
        parsing_all = tf.reshape(parsing_seq, [batch_size * time_length, in_h, in_w, 1])
        
        return tgt_image_all, src_image_all, parsing_all


    def computation_graph(self, tgt_image_all, src_image_all, is_training):
        """Build computation graph."""
        opt = self.opt
        batch_size = opt.batch_size
        sample_per_video = opt.sample_per_video
        time_length = opt.frames_per_video - 1
        _, in_h, in_w, channel = tgt_image_all.get_shape().as_list()

        tgt_end_point = base_net(tgt_image_all, is_training, opt.base_net_weight_reg)
        src_end_point = base_net(src_image_all, is_training, opt.base_net_weight_reg, reuse=True)
        final_tgt_feature_map = tgt_end_point['resnet_v2_50/postnorm']
        # tgt_end_point['resnet_v2_50/block1'] shape /8
        # tgt_end_point['resnet_v2_50/block2'] shape /16
        # tgt_end_point['resnet_v2_50/block3'] shape /32
        # tgt_end_point['resnet_v2_50/block4'] shape /32
        # tgt_end_point['resnet_v2_50/postnorm'] shape /32

        with tf.name_scope('attention_module'):
            local_app_feat, att_mask, multi_scale_mask = attention_net(tgt_end_point, is_training, opt)

        with tf.name_scope('motion_module'):
            global_motion, multi_scale_local_motion, global_mot_feat, local_mot_feat = motion_net(tgt_end_point, src_end_point, att_mask, is_training, opt)

        with tf.name_scope('interaction_module'):
            global_app_feat = tf.reduce_mean(final_tgt_feature_map, [1,2])
            logits = interaction_net(global_app_feat, local_app_feat, global_mot_feat, local_mot_feat, opt)
        
        ############  collect tensors of raw images  ############
        self.tgt_video = tf.reshape(tgt_image_all, [batch_size*sample_per_video, time_length, in_h, in_w, channel])
        self.src_video = tf.reshape(src_image_all, [batch_size*sample_per_video, time_length, in_h, in_w, channel])
        #########  collect tensors of attention_module  #########
        self.att_mask = tf.reshape(att_mask, [batch_size*sample_per_video, time_length, in_h // 32, in_w //32, 1])
        self.full_size_att_mask = tf.reshape(multi_scale_mask[0], [batch_size*sample_per_video, time_length, in_h, in_w, 1])
        ########  collect tensors of motion_module  ########
        self.global_motion = tf.reshape(global_motion, [batch_size*sample_per_video, time_length, -1])
        self.local_motion = tf.reshape(multi_scale_local_motion[0], [batch_size*sample_per_video, time_length, in_h, in_w, -1])
        
        return att_mask, multi_scale_mask, global_motion, multi_scale_local_motion, logits


    def compute_loss(self, scope, tgt_image_seq, src_image_seq, parsing_seq, labels, video_name, is_training=True):
        opt = self.opt
        batch_size = opt.batch_size
        sample_per_video = opt.sample_per_video
        time_length = opt.frames_per_video - 1
        
        tgt_image_all, src_image_all, parsing_all = self.data_preprocessing(tgt_image_seq, src_image_seq, parsing_seq)
        att_mask, multi_scale_mask, global_motion, multi_scale_local_motion, logits = self.computation_graph(tgt_image_all, src_image_all, is_training)
        if not is_training:
            logits = tf.reshape(logits, [opt.batch_size, opt.sample_per_video, opt.num_class])
            logits = tf.reduce_mean(logits, 1)
        
        with tf.name_scope("compute_loss"):
            if opt.cls_loss_weight > 0:
                cls_loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=slim.one_hot_encoding(labels, opt.num_class), weights=opt.cls_loss_weight)
            if opt.parsing_loss_weight > 0:
                mask_parsing_loss, _ = self.compute_mask_parsing_loss(att_mask, parsing_all)
            if opt.pixel_loss_weight > 0:
                pixel_loss = 0.0
            if opt.smooth_loss_weight > 0:
                smooth_loss = 0.0
            
            for s in range(opt.num_scales):
                # smooth_loss for flow
                if opt.smooth_loss_weight > 0:
                    smooth_loss += opt.smooth_loss_weight/(2**s) * self.compute_smooth_loss(multi_scale_local_motion[s])
                if opt.parsing_loss_weight > 0:
                    # print(multi_scale_mask, flush=True)
                    # print(multi_scale_mask[0], flush=True)
                    # print(s, flush=True)
                    # print(multi_scale_mask[s], flush=True)
                    # print(len(multi_scale_mask), flush=True)
                    # print(parsing_all, flush=True)
                    # assert False
                    curr_mask_parsing_loss, curr_parsing = self.compute_mask_parsing_loss(multi_scale_mask[s], parsing_all)
                    mask_parsing_loss += curr_mask_parsing_loss
                
                # Inverse warp the source image to the target image frame
                curr_tgt_image = tf.image.resize_area(tgt_image_all, [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
                curr_src_image = tf.image.resize_area(src_image_all, [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
                curr_proj_image, _ = projective_inverse_warp(curr_src_image, global_motion, multi_scale_local_motion[s], multi_scale_mask[s])
                curr_proj_error = tf.abs(tf.clip_by_value(curr_proj_image,-1,1) - tf.clip_by_value(curr_tgt_image,-1,1))
                if s == 0:
                    src_to_tgt_image = curr_proj_image
                    parsing = curr_parsing
                if opt.pixel_loss_weight > 0:
                    pixel_loss += opt.pixel_loss_weight * tf.reduce_mean(curr_proj_error)
            
            if opt.pixel_loss_weight > 0:
                tf.losses.add_loss(pixel_loss)
            if opt.parsing_loss_weight > 0:
                tf.losses.add_loss(mask_parsing_loss)
            if opt.smooth_loss_weight > 0:
                tf.losses.add_loss(smooth_loss)
            total_loss = tf.add_n(tf.losses.get_losses(scope=scope) + tf.losses.get_regularization_losses(scope=scope), name='total_loss')
        
        self.logits = logits
        self.total_loss = total_loss
        if opt.cls_loss_weight > 0:
            self.cls_loss = cls_loss
        if opt.pixel_loss_weight > 0:
            self.pixel_loss = pixel_loss
        if opt.parsing_loss_weight > 0:
            self.mask_parsing_loss = mask_parsing_loss
        if opt.smooth_loss_weight > 0:
            self.smooth_loss = smooth_loss
        
        self.labels = labels
        self.video_name = video_name
        self.parsing = tf.reshape(parsing, [batch_size*sample_per_video, time_length, opt.img_height, opt.img_width, 1])
        self.src_to_tgt_image = tf.reshape(src_to_tgt_image, [batch_size*sample_per_video, time_length, opt.img_height, opt.img_width, -1])
        
        return total_loss


    def compute_mask_parsing_loss(self, att_mask, parsing):
        opt = self.opt
        batch, height, width, _ = att_mask.get_shape().as_list()
        epsilon = 1e-8
        resize_parsing = tf.image.resize_nearest_neighbor(parsing, [height, width])
        resize_parsing.set_shape([batch, height, width, 1])
        parsing_reg = opt.parsing_loss_weight * tf.reduce_mean(- resize_parsing * tf.log(att_mask + epsilon) - (1 - resize_parsing) * tf.log(1 - att_mask + epsilon))

        return parsing_reg, resize_parsing


    def compute_smooth_loss(self, pred_flow):
        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy
        dx, dy = gradient(pred_flow)
        return tf.reduce_mean(tf.abs(dx)) + \
               tf.reduce_mean(tf.abs(dy))


    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, v in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        if opt.cls_loss_weight > 0:
            tf.summary.scalar("cls_loss", self.cls_loss)
        if opt.pixel_loss_weight > 0:
            tf.summary.scalar("pixel_loss", self.pixel_loss)
        if opt.parsing_loss_weight > 0:
            tf.summary.scalar("mask_parsing_loss", self.mask_parsing_loss)
        if opt.smooth_loss_weight > 0:
            tf.summary.scalar("smooth_loss", self.smooth_loss)


    def postprocessing_image(self, image):
        ## [0,255]
        # _R_MEAN = 123.68
        # _G_MEAN = 116.78
        # _B_MEAN = 103.94
        # mean_vals = np.array([_R_MEAN, _G_MEAN, _B_MEAN])
        # image += mean_vals
        # image = image / 255.0
        ## [0,1]
        image = (image + 1.)/2.
        return image
        
        
    def save(self, sess, checkpoint_dir, step):
        model_name = 'model.ckpt'
        print("Saving checkpoint to %s..." % checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=int(step))


    def visualization(self, step, root_dir, batch_index, video_name, labels, logits, tgt_video, src_video, parsing, src_to_tgt_image, global_motion, att_mask, full_size_att_mask, local_motion, opt):
        dir = os.path.join(root_dir, '{0}_{1}_{2}_{3}'.format(step, video_name[batch_index].decode().replace('/', '-'), labels[batch_index], np.around(logits[batch_index].tolist(),decimals=3).tolist()))
        os.makedirs(dir)
        for t in range(opt.frames_per_video - 1):
            tgt_image_name = os.path.join(dir, '{0}_tgt.jpg'.format(t))
            io.imsave(tgt_image_name, np.clip(self.postprocessing_image(tgt_video[batch_index][t]) , 0, 1))
            src_image_name = os.path.join(dir, '{0}_src.jpg'.format(t))
            io.imsave(src_image_name, np.clip(self.postprocessing_image(src_video[batch_index][t]) , 0, 1))
            parsing_name = os.path.join(dir, '{0}_parsing.jpg'.format(t))
            io.imsave(parsing_name, np.clip(parsing[batch_index][t][:,:,0], 0, 1))
            src_to_tgt_name = os.path.join(dir, '{0}_src_to_tgt_{1}.jpg'.format(t, np.around(global_motion[batch_index][t].tolist(),decimals=3).tolist()))
            io.imsave(src_to_tgt_name, np.clip(self.postprocessing_image(src_to_tgt_image[batch_index][t]) , 0, 1))
            sigmoid_mask_t = att_mask[batch_index][t][:,:,0]
            sigmoid_mask_max = np.max(att_mask)
            sigmoid_mask_min = np.min(att_mask)
            sigmoid_mask_t = (sigmoid_mask_t - sigmoid_mask_min) / (sigmoid_mask_max - sigmoid_mask_min + 1e-8)
            sigmoid_mask_name = os.path.join(dir, '{0}_sigmoid_mask_max_{1:.3}_min_{2:.3}.jpg'.format(t,sigmoid_mask_max,sigmoid_mask_min))
            io.imsave(sigmoid_mask_name, sigmoid_mask_t)
            ori_sigmoid_mask_t = full_size_att_mask[batch_index][t][:,:,0]
            ori_sigmoid_mask_t_max = np.max(ori_sigmoid_mask_t)
            ori_sigmoid_mask_t_min = np.min(ori_sigmoid_mask_t)
            ori_sigmoid_mask_t = (ori_sigmoid_mask_t - ori_sigmoid_mask_t_min) / (ori_sigmoid_mask_t_max - ori_sigmoid_mask_t_min + 1e-8)
            ori_sigmoid_mask_name = os.path.join(dir, '{0}_ori_sigmoid_mask_max_{1:.3}_min_{2:.3}.jpg'.format(t,ori_sigmoid_mask_t_max,ori_sigmoid_mask_t_min))
            io.imsave(ori_sigmoid_mask_name, ori_sigmoid_mask_t)
            local_motion_h = local_motion[batch_index][t][:,:,0]
            h, w = local_motion_h.shape
            assert h == opt.img_height
            assert w == opt.img_width
            local_motion_h = ori_sigmoid_mask_t * local_motion_h * h
            local_motion_h_max = np.max(local_motion_h)
            local_motion_h_min = np.min(local_motion_h)
            local_motion_h = local_motion_h / (2 * np.max(np.abs(local_motion_h)) + 1e-8) + 0.5
            local_motion_h_name = os.path.join(dir, '{0}_local_motion_h_max_{1:.3}_min_{2:.3}.jpg'.format(t,local_motion_h_max,local_motion_h_min))
            io.imsave(local_motion_h_name, local_motion_h)
            local_motion_w = local_motion[batch_index][t][:,:,1]
            h, w = local_motion_w.shape
            assert h == opt.img_height
            assert w == opt.img_width
            local_motion_w = ori_sigmoid_mask_t * local_motion_w * w
            local_motion_w_max = np.max(local_motion_w)
            local_motion_w_min = np.min(local_motion_w)
            local_motion_w = local_motion_w / (2 * np.max(np.abs(local_motion_w)) + 1e-8) + 0.5
            local_motion_w_name = os.path.join(dir, '{0}_local_motion_w_max_{1:.3}_min_{2:.3}.jpg'.format(t,local_motion_w_max,local_motion_w_min))
            io.imsave(local_motion_w_name, local_motion_w)


    def train(self, opt):
        self.opt = opt
        self.gpu_num = len(opt.gpu_list.split(','))
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            self.loader = self.build_dataloader()

            # optim = tf.train.MomentumOptimizer(learning_rate = opt.learning_rate, momentum = 0.9)
            optim = tf.train.AdamOptimizer(learning_rate = opt.learning_rate)
            tower_grads = []
            tower_loss = []
            with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
              for i in range(self.gpu_num):
                with tf.device('/gpu:%d' % i):
                  with tf.variable_scope(tf.get_variable_scope(), reuse=True if i > 0 else None):
                    with tf.name_scope('tower_%d' % i) as sc:
                      tgt_image_seq, src_image_seq, parsing_seq, labels, video_name = self.loader.load_train_batch()
                      loss = self.compute_loss(sc, tgt_image_seq, src_image_seq, parsing_seq, labels, video_name)
                      if i == 0:
                          with tf.name_scope("parameter_count"):
                              self.parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
                          if opt.trainable_scopes is not None:
                              trainable_scopes = opt.trainable_scopes.split(',')
                              trainable_variables = []
                              for var in tf.trainable_variables():
                                  trainable = False
                                  for scope in trainable_scopes:
                                      if var.op.name.startswith(scope):
                                          trainable = True
                                          break
                                  if trainable:
                                      trainable_variables.append(var)
                          else:
                              trainable_variables = tf.trainable_variables()
                      self.collect_summaries()
                      tower_loss.append(loss)
                      grads = optim.compute_gradients(loss, trainable_variables)
                      tower_grads.append(grads)
            print(trainable_variables, flush=True)
            self.loss = tf.reduce_mean(tower_loss)
            grads = self.average_gradients(tower_grads)
            apply_gradient_op = optim.apply_gradients(grads, global_step=self.global_step)
            variable_averages = tf.train.ExponentialMovingAverage(0.98, self.global_step)
            variables_averages_op = variable_averages.apply(trainable_variables)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # update bn in training scopes
            update_ops = []
            for ops in extra_update_ops:
                for scope in trainable_scopes:
                    if scope in ops.name:
                        update_ops.append(ops)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.group(apply_gradient_op, variables_averages_op)

            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            self.summary_op = tf.summary.merge(summaries)

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True), log_device_placement=False))
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print("parameter_count =", sess.run(self.parameter_count))

            save_trainable_variables = tf.trainable_variables()
            global_var = tf.global_variables()
            bn_moving_average_variables = [g for g in global_var if 'moving_mean' in g.name]
            bn_moving_average_variables += [g for g in global_var if 'moving_variance' in g.name]
            saver_variables = save_trainable_variables + bn_moving_average_variables + [self.global_step]
            self.saver = tf.train.Saver(saver_variables, max_to_keep=opt.max_ckpt)
            print(len(saver_variables), flush=True)

            if opt.pretrain_model and not opt.continue_train:
                variables_exclude_scopes = ['attention_net', 'motion_net', 'interaction_net', 'global_step']
                variables_to_restore = []
                for var in saver_variables:
                  excluded = False
                  for exclusion in variables_exclude_scopes:
                    if var.op.name.startswith(exclusion):
                      excluded = True
                      break
                  if not excluded:
                    variables_to_restore.append(var)
                print('Using ImageNet Pretrain Model %s.'%opt.pretrain_model, flush=True)
                print(len(variables_to_restore), flush=True)
                restore_fn = slim.assign_from_checkpoint_fn(opt.pretrain_model, variables_to_restore, ignore_missing_vars=True)
                restore_fn(sess)

            curr_step = 0
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint, flush=True)
                curr_step = int(checkpoint[checkpoint.rindex('-') + 1:])
                self.saver.restore(sess, checkpoint)
            else:
                if opt.attention_checkpoint_file:
                    print("Initialize parameters using parsing loss with checkpoint: %s" % opt.attention_checkpoint_file, flush=True)
                    use_checkpoint_scopes = ['attention_net']
                    variables_to_restore = []
                    for var in saver_variables:
                      for use_scopes in use_checkpoint_scopes:
                        if var.op.name.startswith(use_scopes):
                          variables_to_restore.append(var)
                    print(len(variables_to_restore), flush=True)
                    restore_fn = slim.assign_from_checkpoint_fn(opt.attention_checkpoint_file, variables_to_restore)
                    restore_fn(sess)
                if opt.motion_checkpoint_file:
                    print("Initialize parameters using pixel loss with checkpoint: %s" % opt.motion_checkpoint_file, flush=True)
                    use_checkpoint_scopes = ['motion_net']
                    variables_to_restore = []
                    for var in saver_variables:
                      for use_scopes in use_checkpoint_scopes:
                        if var.op.name.startswith(use_scopes):
                          variables_to_restore.append(var)
                    print(len(variables_to_restore), flush=True)
                    restore_fn = slim.assign_from_checkpoint_fn(opt.motion_checkpoint_file, variables_to_restore)
                    restore_fn(sess)
                if opt.interaction_checkpoint_file:
                    print("Initialize parameters using cls loss with checkpoint: %s" % opt.interaction_checkpoint_file, flush=True)
                    use_checkpoint_scopes = ['interaction_net']
                    variables_to_restore = []
                    for var in saver_variables:
                      for use_scopes in use_checkpoint_scopes:
                        if var.op.name.startswith(use_scopes):
                          variables_to_restore.append(var)
                    print(len(variables_to_restore), flush=True)
                    restore_fn = slim.assign_from_checkpoint_fn(opt.interaction_checkpoint_file, variables_to_restore, ignore_missing_vars=True)
                    restore_fn(sess)

            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
            self.summary_writer = tf.summary.FileWriter(opt.checkpoint_dir, sess.graph)

            start_time = time.time()
            for step in range(curr_step + 1, opt.max_steps + 1):
                fetches = {"train": self.train_op, "global_step": self.global_step, "loss":self.loss}
                if step % opt.summary_freq == 0 or step == opt.max_steps:
                    fetches["summaries"] = self.summary_op
                    if opt.cls_loss_weight > 0:
                        fetches["cls_loss"] = self.cls_loss
                    if opt.pixel_loss_weight > 0:
                        fetches["pixel_loss"] = self.pixel_loss
                    if opt.parsing_loss_weight > 0:
                        fetches["mask_parsing_loss"] = self.mask_parsing_loss
                    if opt.smooth_loss_weight > 0:
                        fetches["smooth_loss"] = self.smooth_loss
                if step == curr_step + 1 or step % opt.vis_freq == 0:
                    fetches['labels'] = self.labels
                    fetches['video_name'] = self.video_name
                    fetches['logits'] = self.logits
                    fetches['tgt_video'] = self.tgt_video
                    fetches['src_video'] = self.src_video
                    fetches['parsing'] = self.parsing
                    fetches['src_to_tgt_image'] = self.src_to_tgt_image
                    fetches['att_mask'] = self.att_mask
                    fetches['full_size_att_mask'] = self.full_size_att_mask
                    fetches['global_motion'] = self.global_motion
                    fetches['local_motion'] = self.local_motion
                results = sess.run(fetches)
                gs = results["global_step"]
                if step % opt.summary_freq == 0 or step == opt.max_steps:
                    if not opt.cls_loss_weight > 0:
                        results["cls_loss"] = 0.0
                    if not opt.pixel_loss_weight > 0:
                        results["pixel_loss"] = 0.0
                    if not opt.parsing_loss_weight > 0:
                        results["mask_parsing_loss"] = 0.0
                    if not opt.smooth_loss_weight > 0:
                        results["smooth_loss"] = 0.0

                if step % opt.summary_freq == 0 or step == opt.max_steps:
                    self.summary_writer.add_summary(results["summaries"], int(gs))
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Total step: [%d] Epoch: [%d] time: %.3f s/step  loss: %.5f\n  cls_loss: %.5f  pixel_loss: %.5f  parsing_loss: %.5f  smooth_loss: %.5f" \
                            % (gs, train_epoch, (time.time() - start_time)/opt.summary_freq, results["loss"], results["cls_loss"], \
                            results["pixel_loss"], results["mask_parsing_loss"], results["smooth_loss"], ), flush=True)
                    start_time = time.time()
                assert not np.isnan(results["loss"])
                
                if opt.vis_flag and step % opt.vis_freq == 0:
                    batch_index = np.random.randint(0, opt.batch_size, [1])[0]
                    self.visualization(step, opt.visual_dir, batch_index, results['video_name'], results['labels'], results['logits'], results['tgt_video'], \
                    results['src_video'], results['parsing'], results['src_to_tgt_image'], results['global_motion'], results['att_mask'], \
                    results['full_size_att_mask'], results['local_motion'], opt)

                # if step % (opt.epoch_per_save * self.steps_per_epoch) == 1 or step == opt.max_steps:
                if step % opt.step_per_save == 1 or step == opt.max_steps:
                    self.save(sess, opt.checkpoint_dir, gs)

        coordinator.request_stop()
        coordinator.join(threads)
        sess.close()
        print("Done.", flush=True)

    
    def inference(self, opt):
        self.opt = opt
        self.gpu_num = len(opt.gpu_list.split(','))
        with tf.Graph().as_default(), tf.device('/gpu:0'):
            with tf.name_scope('tower_0') as sc:
                self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
                self.loader = self.build_dataloader()
                tgt_image_seq, src_image_seq, parsing_seq, labels, video_name = self.loader.load_inference_batch()
                self.loss = self.compute_loss(sc, tgt_image_seq, src_image_seq, parsing_seq, labels, video_name, False)

            self.parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True), log_device_placement=False))
            sess.run(tf.global_variables_initializer())
            print("parameter_count =", sess.run(self.parameter_count))

            save_trainable_variables = tf.trainable_variables()
            global_var = tf.global_variables()
            bn_moving_average_variables = [g for g in global_var if 'moving_mean' in g.name]
            bn_moving_average_variables += [g for g in global_var if 'moving_variance' in g.name]
            saver_variables = save_trainable_variables + bn_moving_average_variables + [self.global_step]
            self.saver = tf.train.Saver(saver_variables)
            print(len(saver_variables), flush=True)

            if opt.checkpoint_file is None:
                checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
            else:
                checkpoint = opt.checkpoint_file
            print("Inference from checkpoint: %s" % checkpoint, flush=True)
            self.saver.restore(sess, checkpoint)

            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

            acc = []
            total_loss = []
            to_obs_loss = []
            for step in range(self.steps_per_epoch):
                fetches = {"global_step": self.global_step}
                fetches["loss"] = self.loss
                fetches['labels'] = self.labels
                fetches['video_name'] = self.video_name
                fetches['logits'] = self.logits
                if step % opt.vis_freq == 0:
                    fetches['tgt_video'] = self.tgt_video
                    fetches['src_video'] = self.src_video
                    fetches['parsing'] = self.parsing
                    fetches['src_to_tgt_image'] = self.src_to_tgt_image
                    fetches['att_mask'] = self.att_mask
                    fetches['full_size_att_mask'] = self.full_size_att_mask
                    fetches['global_motion'] = self.global_motion
                    fetches['local_motion'] = self.local_motion
                if opt.cls_loss_weight > 0:
                    fetches["cls_loss"] = self.cls_loss
                if opt.pixel_loss_weight > 0:
                    fetches["pixel_loss"] = self.pixel_loss
                if opt.parsing_loss_weight > 0:
                    fetches["mask_parsing_loss"] = self.mask_parsing_loss
                if opt.smooth_loss_weight > 0:
                    fetches["smooth_loss"] = self.smooth_loss
                results = sess.run(fetches)
                if not opt.cls_loss_weight > 0:
                    results["cls_loss"] = 0.0
                if not opt.pixel_loss_weight > 0:
                    results["pixel_loss"] = 0.0
                if not opt.parsing_loss_weight > 0:
                    results["mask_parsing_loss"] = 0.0
                if not opt.smooth_loss_weight > 0:
                    results["smooth_loss"] = 0.0
                assert np.shape(results['logits']) == (opt.batch_size, opt.num_class), 'Wrong size of logits.'
                print(results['video_name'], np.argmax(results['logits'], 1), results['labels'], flush=True)
                curr_acc = (np.argmax(results['logits'], 1) == results['labels'])
                acc.append(curr_acc)
                total_loss.append(results["loss"])
                to_obs_loss.append(results[opt.to_obs_loss])

                print("Sample num: [%d] accuracy: %d  loss: %.5f\n  cls_loss: %.5f  pixel_loss: %.5f  mask_parsing_loss: %.5f  smooth_loss: %.5f" \
                        % (step, curr_acc, results["loss"], results["cls_loss"], results["pixel_loss"], results["mask_parsing_loss"], results["smooth_loss"]), flush=True)
                if opt.vis_flag and step % opt.vis_freq == 0:
                    batch_index = np.random.randint(0, opt.batch_size, [1])[0]
                    self.visualization(step, opt.visual_dir, batch_index, results['video_name'], results['labels'], results['logits'], results['tgt_video'], \
                    results['src_video'], results['parsing'], results['src_to_tgt_image'], results['global_motion'], results['att_mask'], \
                    results['full_size_att_mask'], results['local_motion'], opt)

        print('Total accuracy: %.4f.'%np.mean(acc), flush=True)
        print('Total loss: %.4f.'%np.mean(total_loss), flush=True)
        print('%s loss: %.4f.'%(opt.to_obs_loss, np.mean(to_obs_loss)), flush=True)
        coordinator.request_stop()
        coordinator.join(threads)
        sess.close()
        print("Done.", flush=True)
