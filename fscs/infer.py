#!/usr/bin/env python3

import glob
import os

from absl import app
from absl import flags
from absl import logging
from PIL import Image

import numpy as np

import helper


def save_palette_as_image(palette, fn, square_size=64):
    n = len(palette)

    arr = np.zeros((square_size * n, square_size, 3), dtype=np.uint8)
    palette = np.clip(palette, 0, 255).astype(np.uint8)
    for i in range(n):
        for ch in range(3):
            arr[square_size * i:square_size * (i + 1), :, ch] = palette[i, ch]

    img = Image.fromarray(arr)
    img.save(fn)

    img_landscape = Image.fromarray(np.transpose(arr, axes=(1, 0, 2)))
    fn_landscape = os.path.splitext(fn)[0] + '_landscape' + os.path.splitext(fn)[1]
    img_landscape.save(fn_landscape)


def list_dir_images(dir_):
    fns = os.listdir(dir_)
    fns = [fn for fn in fns if fn.endswith('.jpg') or fn.endswith('.png')]
    fns = list(sorted(fns))
    return fns


def single_image_decompose(src_fn, tgt_dir):
    os.makedirs(tgt_dir, exist_ok=True)
    for f in glob.glob(f'{tgt_dir}/*'):
        os.remove(f)

    num_colors = 7
    palette, kmeans = helper.infer_palette(str(src_fn), num_colors=num_colors)
    manual_colors = palette / 255.

    infer = helper.Infer(manual_colors=manual_colors)
    inferred_blob = infer.infer(str(src_fn))
    helper.save_inferred_blob_as_image(inferred_blob, save_dir=tgt_dir)
    save_palette_as_image(palette, f'{tgt_dir}/palette.png')


def main(_):
    FLAGS = flags.FLAGS

    in_image = FLAGS.in_image
    out_dir = FLAGS.out_dir
    assert in_image
    assert out_dir
    logging.info('Applying FSCS to image %s, saving to %s/*', in_image, out_dir)
    single_image_decompose(in_image, out_dir)


if __name__ == '__main__':
    flags.DEFINE_string('in_image', '', '')
    flags.DEFINE_string('out_dir', '', '')
    app.run(main)
