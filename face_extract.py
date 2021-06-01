#!/usr/bin/env python3

# TODO: make it standalone
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import ray

from face_common import (
    ensure_dir_exists,
    get_landmarks_name,
    get_landmark_xy_by_name,
    set_landmark_xy_by_name,
    get_virtualpoint_xy_by_name,
    set_virtualpoint_xy_by_name,
    rel_df_to_abs_df,
    abs_df_to_rel_df,
    radians_to_degree,
    radians_normalize,
    rotate,
    compute_specs,
)


def rotate_and_crop_and_resize(img, specs, size):
    INTERNAL_SCALE_RATIO = 4.0

    l, c, alpha = specs
    l = int(l)
    c = c.astype(np.int32)
    specs = l, c, alpha

    angle = radians_to_degree(radians_normalize(2 * np.pi - alpha))

    center = c.tolist()

    scale_up_size = (int(img.size[0] * INTERNAL_SCALE_RATIO),
                     int(img.size[1] * INTERNAL_SCALE_RATIO))
    scale_down_size = img.size
    scale_up_center = (
        int(center[0] * INTERNAL_SCALE_RATIO),
        int(center[1] * INTERNAL_SCALE_RATIO),
    )
    img = img.resize(scale_up_size, resample=Image.LANCZOS)
    img = img.rotate(angle=angle, center=scale_up_center)
    img = img.resize(scale_down_size, resample=Image.LANCZOS)
    result_img = img

    left = c[0] - l // 2
    right = c[0] - l // 2 + l
    upper = c[1] - l // 2
    lower = c[1] - l // 2 + l

    result_img = result_img.crop([left, upper, right, lower])

    result_img = result_img.resize((size, size), resample=Image.LANCZOS)

    return result_img


def compute_new_row(abs_row, specs, new_size):
    l, c, alpha = specs

    def get_new_xy(xy):
        new_xy = rotate(xy - c, alpha) + l / 2
        new_xy = np.clip(new_xy, 0.0, l + 1e-8)
        new_xy = new_xy / l * new_size
        return new_xy

    new_abs_row = abs_row.copy()
    new_abs_row['width'] = new_size
    new_abs_row['height'] = new_size
    new_abs_row['filename'] = new_abs_row['singleface_filename']

    for landmark_name in get_landmarks_name(new_abs_row):
        xy = get_landmark_xy_by_name(new_abs_row, landmark_name, as_array=True)
        new_xy = get_new_xy(xy=xy)
        set_landmark_xy_by_name(new_abs_row, landmark_name, new_xy)

    top_left_xy = get_virtualpoint_xy_by_name(new_abs_row, 'top_left')
    bottom_right_xy = get_virtualpoint_xy_by_name(new_abs_row, 'bottom_right')
    set_virtualpoint_xy_by_name(new_abs_row, 'top_left',
                                get_new_xy(xy=top_left_xy))
    set_virtualpoint_xy_by_name(new_abs_row, 'bottom_right',
                                get_new_xy(xy=bottom_right_xy))

    return new_abs_row


def go(abs_row, new_size, images_dir, new_images_dir):
    specs = compute_specs(abs_row)
    new_abs_row = compute_new_row(abs_row, specs, new_size)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(os.path.join(images_dir, abs_row['filename']))
    result_img = rotate_and_crop_and_resize(img, specs, new_size)
    result_img.save(os.path.join(new_images_dir, new_abs_row['filename']))

    return new_abs_row


@ray.remote
def go_ray_actor(*args, **kwargs):
    return go(*args, **kwargs)


def main(_):
    FLAGS = flags.FLAGS

    images_dir = FLAGS.images_dir
    face_landmarks_file = FLAGS.face_landmarks_file
    arc_metadata_file = FLAGS.arc_metadata_file
    new_images_dir = FLAGS.new_images_dir
    new_face_landmarks_file = FLAGS.new_face_landmarks_file
    new_arc_metadata_file = FLAGS.new_arc_metadata_file
    new_size = FLAGS.new_size
    parallel_compute = FLAGS.parallel_compute
    max_rows = FLAGS.max_rows

    face_landmarks = pd.read_csv(face_landmarks_file)

    if os.path.exists(arc_metadata_file):
        arc_metadata = pd.read_csv(arc_metadata_file)
        _df = arc_metadata.join(face_landmarks[['filename', 'singleface_filename'
                                                ]].set_index('filename'),
                                on='filename')
        _df = _df[pd.notna(_df['singleface_filename'])]
        _df = _df.drop(columns=['filename']).rename(
            columns={'singleface_filename': 'filename'})
        ensure_dir_exists(new_arc_metadata_file)
        _df.to_csv(new_arc_metadata_file, index=False)

    abs_df = rel_df_to_abs_df(face_landmarks)
    os.makedirs(new_images_dir, exist_ok=True)
    if max_rows <= 0:
        max_rows = len(abs_df)
    if parallel_compute:
        ray.shutdown()
        num_cpus = FLAGS.num_cpus if FLAGS.num_cpus >= 1 else None
        ray.init(num_cpus=num_cpus)
        futures = [
            go_ray_actor.remote(
                abs_df.iloc[row_index],
                new_size,
                images_dir,
                new_images_dir,
            ) for row_index in range(max_rows)
        ]
        logging.info(f'Paralleing processing: {len(futures)} tasks running.')
        new_abs_rows = ray.get(futures)
        logging.info(f'Paralleing processing: {len(futures)} tasks done.')
    else:
        new_abs_rows = [
            go(abs_df.iloc[row_index], new_size, images_dir, new_images_dir)
            for row_index in range(max_rows)
        ]

    new_abs_df = pd.DataFrame.from_records(new_abs_rows)
    new_df = abs_df_to_rel_df(new_abs_df)
    ensure_dir_exists(new_face_landmarks_file)
    new_df.to_csv(new_face_landmarks_file, index=False)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('images_dir', '', '')
    flags.DEFINE_string('face_landmarks_file', '', '')
    flags.DEFINE_string('arc_metadata_file', '', '')
    flags.DEFINE_string('new_images_dir', '', '')
    flags.DEFINE_string('new_face_landmarks_file', '', '')
    flags.DEFINE_string('new_arc_metadata_file', '', '')
    flags.DEFINE_integer('new_size', 512, '')
    flags.DEFINE_boolean('parallel_compute', True, '')
    flags.DEFINE_integer('max_rows', 0, '')
    flags.DEFINE_integer(
        'num_cpus', -1,
        'Number of CPUs to use in parallel. If less than 1 (including the default value -1), use all CPUs.'
    )

    app.run(main)
