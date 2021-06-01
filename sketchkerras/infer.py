#!/usr/bin/env python3

import os

from absl import app
from absl import flags
from absl import logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

import helper


def convert(path_in, path_out, mod, style='default'):
    from_mat = cv2.imread(path_in)
    width = float(from_mat.shape[1])
    height = float(from_mat.shape[0])
    new_width = 0
    new_height = 0
    if (width > height):
        from_mat = cv2.resize(from_mat, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
        new_width = 512
        new_height = int(512 / width * height)
    else:
        from_mat = cv2.resize(from_mat, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
        new_width = int(512 / height * width)
        new_height = 512
    from_mat = from_mat.transpose((2, 0, 1))
    light_map = np.zeros(from_mat.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = helper.get_light_map_single(from_mat[channel])
    light_map = helper.normalize_pic(light_map)
    light_map = helper.resize_img_512_3d(light_map)

    line_mat = mod.predict(light_map, batch_size=1)
    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
    line_mat = np.amax(line_mat, 2)
    line_mat = cv2.resize(line_mat, (int(width), int(height)), interpolation=cv2.INTER_AREA)

    save_func = {
        'default': helper.show_active_img_and_save_denoise,
        'pured': helper.show_active_img_and_save_denoise_filter,
        'enhanced': helper.show_active_img_and_save_denoise_filter2,
    }[style]
    save_func('_', line_mat, path_out)


def main(_):
    FLAGS = flags.FLAGS

    in_image = FLAGS.in_image
    out_image = FLAGS.out_image
    model_path = FLAGS.model_path

    logging.info('Extract like art for images in %s, saving to %s' % (in_image, out_image))
    mod = tf.keras.models.load_model(model_path)
    os.makedirs(os.path.dirname(out_image), exist_ok=True)

    convert(in_image, out_image, mod)


if __name__ == "__main__":
    flags.DEFINE_string('in_image', '', '')
    flags.DEFINE_string('out_image', '', '')
    flags.DEFINE_string('model_path', f'{os.path.dirname(__file__)}/mod.h5', '')
    app.run(main)