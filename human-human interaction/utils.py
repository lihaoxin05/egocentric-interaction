from __future__ import division
import numpy as np
import tensorflow as tf


def translation_vec2mat(vec):
    """Converts translation parameters to transformation matrix
    Args:
      vec: translation parameters in the order of x, y -- [B, 2]
    Returns:
      A transformation matrix -- [B, 3, 3]
    """
    batch_size, vec_size = vec.get_shape().as_list()
    identity = tf.eye(vec_size, batch_shape=[batch_size])
    vec = tf.expand_dims(vec, -1)
    zeros_vec = tf.zeros([batch_size, 1, vec_size])
    ones_vec = tf.ones([batch_size, 1, 1])
    mat = tf.concat([tf.concat([identity, vec], 2), tf.concat([zeros_vec, ones_vec], 2)], 1)
    return mat


def angle2mat(r):
    """Converts rotation parameters to transformation matrix
    Args:
      r: rotation parameters in the order of x, y -- [B, 1]
    Returns:
      A transformation matrix -- [B, 3, 3]
    """
    batch_size, vec_size = r.get_shape().as_list()
    r = tf.expand_dims(r, -1)
    zeros = tf.zeros([batch_size, vec_size, 1])
    ones  = tf.ones([batch_size, vec_size, 1])

    cosr = tf.cos(r)
    sinr = tf.sin(r)
    rotr_1 = tf.concat([cosr, -sinr], axis=2)
    rotr_2 = tf.concat([sinr, cosr], axis=2)
    rotMat = tf.concat([rotr_1, rotr_2], axis=1)
    rotMat = tf.concat([rotMat, tf.concat([zeros, zeros], 1)], 2)
    rotMat = tf.concat([rotMat, tf.concat([zeros, zeros, ones], 2)], 1)
    return rotMat


def pose_vec2mat(vec):
    """Converts pose parameters to transformation matrix
    Args:
      vec: pose parameters in the order of tx, ty, rx, ry, r, sx0, sy0, sx, sy -- [B, 9]
    Returns:
      A transformation matrix -- [B, 3, 3]
    """
    batch_size, _ = vec.get_shape().as_list()
    translation = tf.slice(vec, [0, 0], [-1, 2])
    translation = tf.nn.tanh(translation)
    r_center = tf.slice(vec, [0, 2], [-1, 2])
    r_center = tf.nn.tanh(r_center)
    inverse_r_center = -r_center
    r = tf.slice(vec, [0, 4], [-1, 1])
    r = np.pi * tf.nn.tanh(r)
    scale_center = tf.slice(vec, [0, 5], [-1, 2])
    scale_center = tf.nn.tanh(scale_center)
    inverse_scale_center = -scale_center
    scale = tf.slice(vec, [0, 7], [-1, 2])
    scale = tf.exp(tf.concat([scale, tf.zeros([batch_size, 1])], 1))

    translation_mat = translation_vec2mat(translation)
    r_center_mat = translation_vec2mat(r_center)
    inverse_r_center_mat = translation_vec2mat(inverse_r_center)
    scale_center_mat = translation_vec2mat(scale_center)
    inverse_scale_center_mat = translation_vec2mat(inverse_scale_center)
    rot_mat = angle2mat(r)
    diag = lambda x: tf.diag(x)
    scale_mat =  tf.map_fn(diag, scale)

    transform_mat = tf.matmul(translation_mat, inverse_r_center_mat)
    transform_mat = tf.matmul(transform_mat, rot_mat)
    transform_mat = tf.matmul(transform_mat, r_center_mat)
    transform_mat = tf.matmul(transform_mat, inverse_scale_center_mat)
    transform_mat = tf.matmul(transform_mat, scale_mat)
    transform_mat = tf.matmul(transform_mat, scale_center_mat)
    return transform_mat


def pixel2pixel(pixel_coords, global_proj, local_proj, mask):
    """Transforms coordinates in a camera frame to the pixel frame.
    Args:
        pixel_coords: [batch, 3, height, width]
        global_proj: [batch, 3, 3]
        local_proj:[batch, 3, height, width]
        mask: [batch, height, width, 1]
    Returns:
        Pixel coordinates projected from the camera frame [batch, height, width, 2]
        3D coordinates projected from the camera frame [batch, height, width, 3]
    """
    batch, _, height, width = pixel_coords.get_shape().as_list()
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    local_proj = tf.reshape(local_proj, [batch, 3, -1])
    mask = tf.transpose(mask, perm=[0, 3, 1, 2])
    mask = tf.reshape(mask, [batch, 1, -1])
    mask =  tf.clip_by_value(mask, 0, 1)
    mask = tf.round(mask)
    
    local_motion_cam_coords = mask * local_proj + pixel_coords
    unnormalized_pixel_coords = tf.matmul(global_proj, local_motion_cam_coords)
    proj_cam_coords = tf.reshape(unnormalized_pixel_coords, [batch, -1, height, width])
    ## normalization
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    x_n = (x_u + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_n = (y_u + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    pixel_coords = tf.concat([x_n, y_n], axis=1)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
    return tf.transpose(pixel_coords, perm=[0, 2, 3, 1]), tf.transpose(proj_cam_coords, perm=[0, 2, 3, 1])


def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.
    Args:
        batch: batch size
        height: height of the grid
        width: width of the grid
        is_homogeneous: whether to return homogeneous coordinates
    Returns:
        x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])), tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1), tf.ones(shape=tf.stack([1, width])))
    if is_homogeneous:
        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=0)
    else:
        coords = tf.stack([x_t, y_t], axis=0)
    coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords


def projective_inverse_warp(img, global_motion, local_motion, att_mask):
    """Inverse warp a source image to the target image plane based on projection.
    Args:
        img: the source image -- [batch, height_s, width_s, 3]
        global_motion: global motion parameters in the order of tx, ty, rx, ry, r, sx0, sy0, sx, sy -- [batch, 9]
        local_motion: local motion field -- [batch, height_s, width_s, 2]
        att_mask: local attention mask -- [batch, height_s, width_s, 1]
    Returns:
        Source image inverse warped to the target image plane [batch, height_t, width_t, 3]
    """
    batch, height, width, _ = img.get_shape().as_list()
    ##### set motion parameters for debug #####
    # global_motion = tf.zeros_like(global_motion)
    # # global_motion = tf.tile(tf.constant([0.0, 0.0, 0.0, 0.1, 0.1, 0.0], shape=[1,6]),[batch, 1])
    # local_motion = tf.zeros_like(local_motion)
    #######################################
    ## Convert global motion vector to matrix
    global_motion = pose_vec2mat(global_motion)
    ## local motion
    zeros_motion = tf.zeros([batch, 1, height, width])
    local_motion = tf.transpose(local_motion, [0,3,1,2])
    local_motion = tf.concat([local_motion, zeros_motion], axis=1)
    ## Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)
    ## coordinates transformation
    tgt_cam_coords = pixel_coords
    src_cam_coords = pixel_coords
    proj_src_pixel_coords, proj_src_cam_coords = pixel2pixel(tgt_cam_coords, global_motion, local_motion, att_mask)

    output_img = bilinear_sampler(img, proj_src_pixel_coords)
    return output_img, [src_cam_coords, proj_src_cam_coords]


def bilinear_sampler(imgs, coords):
    """Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value 0.

    Args:
        imgs: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,
          width_t, 2]. height_t/width_t correspond to the dimensions of the output
          image (don't need to be the same as height_s/width_s). The two channels
          correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels]
    """
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats,])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        ## bilinear interp weights, with points outside the grid having weight 0
        wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
        wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
        wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
        wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = tf.reshape(x0_safe + base_y1, [-1])
        idx10 = tf.reshape(x1_safe + base_y0, [-1])
        idx11 = tf.reshape(x1_safe + base_y1, [-1])

        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, tf.int32)), out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, tf.int32)), out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, tf.int32)), out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, tf.int32)), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        return output
    