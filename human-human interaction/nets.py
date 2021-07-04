from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import resnet_utils
from InteractiveLSTM import *
resnet_arg_scope = resnet_utils.resnet_arg_scope


def resnet_v2_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v2 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v2 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)

def resnet_v2(inputs,
              blocks,
              is_training=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None):

  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense,
                         slim.batch_norm],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4

          with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          with slim.arg_scope([slim.max_pool2d], kernel_size=[3,3], stride=2):
            net = slim.max_pool2d(net, scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)

        net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, scope='postnorm')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        return end_points

        
def resnet_v2_50(inputs,
                 is_training=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_50'):
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return resnet_v2(inputs, blocks, is_training=is_training,
                     output_stride=output_stride,
                     include_root_block=True,
                     reuse=reuse, scope=scope)

def base_net(input, is_training, weight_decay, reuse=False):
    with slim.arg_scope(resnet_arg_scope(weight_decay=weight_decay)):
        return resnet_v2_50(input, is_training=is_training, reuse=reuse)


def resize_like(inputs, ref):
    """Resize the image or feature map to the size of reference image or feature map.
    Args:
        inputs: A tensor of size [in_batch, in_height, in_width, in_channels].
        ref: The reference tensor of size [ref_batch, ref_height, ref_width, ref_channels].

    Returns:
        The resized tensor of size [in_batch, ref_height, ref_width, in_channels].
    """
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


def attention_net(end_point, is_training, opt):
    """Generate attention masks and attention feature maps.
    Args:
        end_point: base net feature maps.
        is_training: training flag.
        opt: parameters.
    Returns:
        attention features.
        masks of different scalse.
    """
    
    img_height, img_width, weight_decay, num_scales = opt.img_height, opt.img_width, opt.attention_net_weight_reg, opt.num_scales
    
    mid_feature_map_1 = end_point['resnet_v2_50/conv1']
    mid_feature_map_2 = end_point['resnet_v2_50/block1']
    final_feature_map = end_point['resnet_v2_50/postnorm']
    batch_size, final_h, final_w, in_channel = final_feature_map.get_shape().as_list()
    ############# mask_generation #############
    with tf.variable_scope('attention_net'):
        with tf.variable_scope('mask'):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], kernel_size=3, stride=1,
                                 normalizer_fn=None,
                                 weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
                                 weights_regularizer=slim.l2_regularizer(weight_decay),
                                 activation_fn=tf.nn.leaky_relu):
                mask = slim.conv2d(final_feature_map, num_outputs=512, scope='conv1')
                mask = slim.conv2d(mask, num_outputs=128, scope='conv2')
                mask = slim.conv2d(mask, num_outputs=32, scope='conv3')
                sigmoid_mask = slim.conv2d(mask, num_outputs=1, activation_fn=tf.nn.sigmoid, scope='conv4')
                
                mask3 = tf.image.resize_bilinear(sigmoid_mask, [img_height // (2**2), img_width // (2**2)])
                upcnv3 = slim.conv2d_transpose(final_feature_map, 512, stride=2, scope='deconv_3_1')
                upcnv3 = slim.conv2d_transpose(upcnv3, 128, stride=2, scope='deconv_3_2')
                upcnv3 = slim.conv2d(upcnv3, num_outputs=64, scope='conv5')
                upcnv3 = resize_like(upcnv3, mask3)
                mask3 = tf.concat([mask3, upcnv3], -1)
                mask3 = slim.conv2d(mask3, num_outputs=1, activation_fn=tf.nn.sigmoid, scope='mask3')
                
                mask2 = tf.image.resize_bilinear(mask3, [img_height // (2**1), img_width // (2**1)])
                upcnv2 = slim.conv2d_transpose(mid_feature_map_2, 128, stride=2, scope='deconv_2_1')
                upcnv2 = slim.conv2d_transpose(upcnv2, 32, stride=2, scope='deconv_2_2')
                upcnv2 = slim.conv2d(upcnv2, num_outputs=8, scope='conv6')
                upcnv2 = resize_like(upcnv2, mask2)
                upcnv2_1 = slim.conv2d_transpose(upcnv3, 32, stride=2, scope='deconv_2_3')
                upcnv2_1 = slim.conv2d(upcnv2_1, num_outputs=16, scope='conv7')
                upcnv2_1 = resize_like(upcnv2_1, mask2)
                mask2 = tf.concat([mask2, upcnv2, upcnv2_1], -1)
                mask2 = slim.conv2d(mask2, num_outputs=1, activation_fn=tf.nn.sigmoid, scope='mask2')
                
                mask1 = tf.image.resize_bilinear(mask2, [img_height // (2**0), img_width // (2**0)])
                upcnv1 = slim.conv2d_transpose(mid_feature_map_1, 32, stride=2, scope='deconv_1_1')
                upcnv1 = slim.conv2d(upcnv1, num_outputs=8, scope='conv8')
                upcnv1 = resize_like(upcnv1, mask1)
                upcnv1_1 = slim.conv2d_transpose(upcnv2_1, 16, stride=2, scope='deconv_1_2')
                upcnv1_1 = slim.conv2d(upcnv1_1, num_outputs=8, scope='conv9')
                upcnv1_1 = resize_like(upcnv1_1, mask1)
                upcnv1_1 = resize_like(upcnv1_1, mask1)
                mask1 = tf.concat([mask1, upcnv1, upcnv1_1], -1)
                mask1 = slim.conv2d(mask1, num_outputs=1, activation_fn=tf.nn.sigmoid, scope='mask1')

        with tf.variable_scope('feature'):
            att_feature = tf.reduce_sum(tf.multiply(final_feature_map, sigmoid_mask), [1,2]) / (tf.reduce_sum(sigmoid_mask, [1,2]) + 1e-8)
    
    return att_feature, sigmoid_mask, [mask1, mask2, mask3]


def motion_net(tgt_end_point, src_end_point, att_mask, is_training, opt):
    """Genarate global and local motion parameters and motion features.
    Args:
        tgt_end_point, src_end_point: source and target feature maps.
        att_mask: attention mask.
        is_training: training flags.
        opt: parameters.
    Returns:
        motion parameters
        motion features
    """
    NEIGHBORHOOD_SIZE, STRIDE, stack_feat_size, out_h, out_w, weight_decay = opt.NEIGHBORHOOD_SIZE, opt.STRIDE, opt.stack_feat_size, opt.img_height, opt.img_width, opt.motion_net_weight_reg
    MAX_DISPLACEMENT = int(math.ceil(NEIGHBORHOOD_SIZE / 2.0))
    global_motion_para_size = 9
    flow_channel = 2
    
    tgt_feature_map = tgt_end_point['resnet_v2_50/block2']
    src_feature_map = src_end_point['resnet_v2_50/block2']
    batch_size, in_h, in_w, in_channel = tgt_feature_map.get_shape().as_list()

    with tf.variable_scope('motion_net'):
        ########### feature concatetnation and correlation ############
        comb_concat = tf.concat([tgt_feature_map, src_feature_map], -1)
        out = []
        for i in range(-MAX_DISPLACEMENT + 1, MAX_DISPLACEMENT, STRIDE): # height
            for j in range(-MAX_DISPLACEMENT + 1, MAX_DISPLACEMENT, STRIDE): # width
                tgt_padding_height = [0, i] if i >=0 else [-i, 0]
                tgt_padding_width = [0, j] if j >=0 else [-j, 0]
                src_padding_height = [i, 0] if i >=0 else [0, -i]
                src_padding_width = [j, 0] if j >=0 else [0, -j]
                padded_tgt_feature = tf.pad(tgt_feature_map, [[0, 0], tgt_padding_height, tgt_padding_width, [0, 0]])
                padded_src_feature = tf.pad(src_feature_map, [[0, 0], src_padding_height, src_padding_width, [0, 0]])
                m = padded_tgt_feature * padded_src_feature

                height_start_idx = src_padding_height[0]
                height_end_idx = height_start_idx + in_h
                width_start_idx = src_padding_width[0]
                width_end_idx = width_start_idx + in_w
                cut = m[:, height_start_idx:height_end_idx, width_start_idx:width_end_idx, :]

                final = tf.sqrt(tf.reduce_mean(cut, 3))
                out.append(final)
        corr = tf.stack(out, 3)
        with slim.arg_scope([slim.batch_norm], center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training):
            corr = slim.batch_norm(corr, scope='batchnorm_corr')
            comb_concat = slim.batch_norm(comb_concat, scope='batchnorm_concat')
            total_feature = tf.concat([corr, comb_concat], -1)

        ########## feature convolution #################
        with slim.arg_scope([slim.conv2d], kernel_size=1, padding='SAME', stride=1, activation_fn=None,
                      weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
            total_feature = slim.conv2d(total_feature, num_outputs=stack_feat_size, scope='conv_total')
            total_feature = slim.batch_norm(total_feature, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, scope='batchnorm_total')
        
        ########### global transformation feature ####################
        with slim.arg_scope([slim.avg_pool2d], kernel_size=[in_h, in_w], stride=1):
            global_motion_feat = tf.squeeze(slim.avg_pool2d(total_feature, scope='global_pool'))
        with slim.arg_scope([slim.fully_connected], activation_fn=None, 
                      weights_initializer=tf.zeros_initializer(),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
            global_para = slim.fully_connected(global_motion_feat, global_motion_para_size, scope='fc_global')
        
        ########## local dense flow generation #######################
        _, h, w, _ = total_feature.shape
        att_mask = tf.image.resize_bilinear(att_mask, [h, w])
        local_motion_feat = tf.multiply(total_feature, att_mask) / tf.reduce_mean(att_mask, [1,2], keep_dims=True)
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            kernel_size=3,
                            normalizer_fn=None,
                            weights_initializer=tf.random_normal_initializer(stddev=1e-3),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            activation_fn=None):
            cnv_feat = slim.conv2d(local_motion_feat, 64, stride=1, scope='cnv_feat')
            cnv_feat = slim.batch_norm(cnv_feat, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, scope='bn_cnv_feat')
            local_motion_feat = tf.reduce_mean(local_motion_feat, [1,2])
            
            upcnv3 = slim.conv2d_transpose(cnv_feat, 32, stride=2, scope='upcnv3')
            upcnv3 = tf.image.resize_bilinear(upcnv3, [out_h // (2**2), out_w // (2**2)])
            upcnv3 = slim.batch_norm(upcnv3, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, scope='bn_upcnv3')
            flow3 = slim.conv2d(upcnv3, flow_channel, stride=1, activation_fn=tf.nn.tanh, scope='flow3')
            
            flow3_up = tf.image.resize_bilinear(flow3, [out_h // 2, out_w // 2])
            upcnv2 = slim.conv2d_transpose(upcnv3, 32, stride=2, scope='upcnv2')
            upcnv2 = resize_like(upcnv2, flow3_up)
            upcnv2 = slim.batch_norm(upcnv2, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, scope='bn_upcnv2')
            i2_in = tf.concat([upcnv2, flow3_up], axis=3)
            icnv2 = slim.conv2d(i2_in, 16, stride=1, scope='icnv2')
            icnv2 = slim.batch_norm(icnv2, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, scope='bn_icnv2')
            flow2 = slim.conv2d(icnv2, flow_channel, stride=1, activation_fn=tf.nn.tanh, scope='flow2')
            
            flow2_up = tf.image.resize_bilinear(flow2, [out_h, out_w])
            upcnv1 = slim.conv2d_transpose(icnv2, 16, stride=2, scope='upcnv1')
            upcnv1 = resize_like(upcnv1, flow2_up)
            upcnv1 = slim.batch_norm(upcnv1, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, scope='bn_upcnv1')
            i1_in = tf.concat([upcnv1, flow2_up], axis=3)
            icnv1 = slim.conv2d(i1_in, 8, stride=1, scope='icnv1')
            icnv1 = slim.batch_norm(icnv1, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, scope='bn_icnv1')
            flow1 = slim.conv2d(icnv1, flow_channel, stride=1, activation_fn=tf.nn.tanh, scope='flow1')

    return global_para, [flow1, flow2, flow3], global_motion_feat, local_motion_feat


def interaction_net(global_app_feat, local_app_feat, global_mot_feat, local_mot_feat, opt):
    """Modeling interaction and classification.
    Args:
        global_app_feat: global appearance features.
        local_app_feat: local appearance features.
        global_mot_feat: global motion features.
        local_mot_feat: local motion features.
        opt: parameters.
    Returns:
        logits: classification logits.
    """
    batch_size, time_length, hidden_size, num_class, feat_size, wd, kp = opt.batch_size*opt.sample_per_video, opt.frames_per_video - 1, opt.hidden_size, opt.num_class, opt.stack_feat_size, opt.interaction_net_weight_reg, opt.keep_prob

    with tf.variable_scope('interaction_net'):
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu, 
                          weights_initializer=tf.initializers.random_normal(0.0, 1e-4),
                          weights_regularizer=tf.contrib.layers.l2_regularizer(wd),
                          trainable=False):
            global_app_feat = slim.fully_connected(global_app_feat, feat_size, scope='embedding_global_app')
            local_app_feat = slim.fully_connected(local_app_feat, feat_size, scope='embedding_local_app')
            
        ego_feature = tf.concat([global_app_feat, global_mot_feat], axis=-1)
        ego_feature = tf.reshape(ego_feature, [batch_size, time_length, -1])
        exo_feature = tf.concat([local_app_feat, local_mot_feat], axis=-1)
        exo_feature = tf.reshape(exo_feature, [batch_size, time_length, -1])
        total_feature = tf.concat([ego_feature, exo_feature], axis=-1)
        # interactive_feature = tf.reduce_mean(ego_feature, 1)
        
        # with tf.variable_scope('total_lstm'):
            # total_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            # total_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(total_lstm_cell, output_keep_prob=kp)
            # initial_state = total_lstm_cell.zero_state(batch_size, dtype=tf.float32)
            # outputs, state = tf.nn.dynamic_rnn(total_lstm_cell, total_feature, initial_state=initial_state, dtype=tf.float32)
            # interactive_feature = outputs[:,-1,:]
        
        # with tf.variable_scope('ego_lstm'):
            # ego_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            # ego_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(ego_lstm_cell, output_keep_prob=kp)
            # initial_state = ego_lstm_cell.zero_state(batch_size, dtype=tf.float32)
            # outputs, state = tf.nn.dynamic_rnn(ego_lstm_cell, ego_feature, initial_state=initial_state, dtype=tf.float32)
            # interactive_feature = outputs[:,-1,:]
            
        # with tf.variable_scope('exo_lstm'):
            # exo_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            # exo_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(exo_lstm_cell, output_keep_prob=kp)
            # initial_state = exo_lstm_cell.zero_state(batch_size, dtype=tf.float32)
            # outputs, state = tf.nn.dynamic_rnn(exo_lstm_cell, exo_feature, initial_state=initial_state, dtype=tf.float32)
            # interactive_feature = outputs[:,-1,:]
        
        with tf.variable_scope('interactive_lstm'):
            interactive_lstm_cell = InteractiveLSTMCell(hidden_size, regularizer_scale=wd)
            interactive_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(interactive_lstm_cell, output_keep_prob=kp)
            initial_state = interactive_lstm_cell.zero_state(batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(interactive_lstm_cell, total_feature, initial_state=initial_state, dtype=tf.float32)
            # interactive_feature = outputs[:,-1,:]
            
        ego_state, exo_state = tf.split(outputs, 2, axis=-1)
        dual_state = tf.tanh((ego_state + exo_state) / 2)
        with tf.variable_scope('total_lstm'):
            total_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            total_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(total_lstm_cell, output_keep_prob=kp)
            initial_state = total_lstm_cell.zero_state(batch_size, dtype=tf.float32)
            outputs, state = tf.nn.dynamic_rnn(total_lstm_cell, dual_state, initial_state=initial_state, dtype=tf.float32)
            interactive_feature = outputs[:,-1,:]
            
        with slim.arg_scope([slim.fully_connected], activation_fn=None, 
                      weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG', uniform=False),
                      weights_regularizer=slim.l2_regularizer(wd)):
            interactive_feature = slim.dropout(interactive_feature, kp, scope='Dropout')
            logits = slim.fully_connected(interactive_feature, num_class, scope='logit')
    return logits
