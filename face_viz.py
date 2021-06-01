#!/usr/bin/env python

import os
import sys

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile
import ray

from face_common import (
    change_extension,
    get_landmarks_name,
    get_landmark_xy_by_name,
    rel_df_to_abs_df,
    rotate,
    compute_specs,
)


def face_viz(face_landmarks_df_or_series, images_dir, out_images_dir, force,
             draw_bounding_box, draw_inferred_box):
    # face_landmarks_df_or_series 's coodinates must be absulute.

    if isinstance(face_landmarks_df_or_series, pd.DataFrame):
        for _index, row in face_landmarks_df_or_series.iterrows():
            face_viz(row, images_dir, out_images_dir, force, draw_bounding_box,
                     draw_inferred_box)
        return

    assert isinstance(face_landmarks_df_or_series, pd.Series)
    row = face_landmarks_df_or_series

    singleface_filename = change_extension(row['singleface_filename'], '.png')
    out_fp = os.path.join(out_images_dir, singleface_filename)
    if not force and os.path.exists(out_fp):
        return  # Skip.

    # Preparing drawing
    fp = os.path.join(images_dir, row['filename'])
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(fp)
    draw = ImageDraw.Draw(img)
    COLOR = (255, 0, 0)  # Red
    LINE_WIDTH = 4
    DOT_WIDTH = 4

    if draw_inferred_box:
        abs_row = row
        specs = compute_specs(abs_row)
        l, c, alpha = specs

        box_vertices = [
            c + rotate(vertex, -alpha) for vertex in [
                (l / 2., l / 2.),
                (l / 2., -l / 2.),
                (-l / 2., -l / 2.),
                (-l / 2., l / 2.),
            ]
        ]
        xy = [tuple(_.astype(np.int32).tolist()) for _ in box_vertices]

        draw.line(xy + xy[:1], fill=COLOR, width=LINE_WIDTH)

    # Draw landmarks
    for landmark_name in get_landmarks_name(row):
        x, y = get_landmark_xy_by_name(row, landmark_name)
        draw.line([(x - DOT_WIDTH // 2, y), (x + DOT_WIDTH // 2 + 1, y)],
                  fill=COLOR,
                  width=DOT_WIDTH // 2)
        draw.line([(x, y - DOT_WIDTH // 2), (x, y + DOT_WIDTH // 2 + 1)],
                  fill=COLOR,
                  width=DOT_WIDTH // 2)

    # Save image
    img.save(out_fp)


@ray.remote
def face_viz_ray_actor(*args, **kwargs):
    face_viz(*args, **kwargs)


def main(_):
    FLAGS = flags.FLAGS
    images_dir = FLAGS.images_dir
    out_images_dir = FLAGS.out_images_dir
    face_landmarks_file = FLAGS.face_landmarks_file
    max_images = FLAGS.max_images
    parallel_compute = FLAGS.parallel_compute
    force = FLAGS.force
    draw_inferred_box = FLAGS.draw_inferred_box

    face_landmarks = pd.read_csv(face_landmarks_file)
    os.makedirs(out_images_dir, exist_ok=True)
    abs_df = rel_df_to_abs_df(face_landmarks)
    if max_images > 0:
        abs_df = abs_df[:max_images]

    if parallel_compute:
        ray.shutdown()
        num_cpus = FLAGS.num_cpus if FLAGS.num_cpus >= 1 else None
        ray.init(num_cpus=num_cpus)
        futures = []
        for _, row in abs_df.iterrows():
            futures.append(
                face_viz_ray_actor.remote(
                    row,
                    images_dir=images_dir,
                    out_images_dir=out_images_dir,
                    force=force,
                    draw_inferred_box=draw_inferred_box,
                ))
        logging.info(f'Paralleing processing: {len(futures)} tasks running.')
        _ = ray.get(futures)
        logging.info(f'Paralleing processing: {len(futures)} tasks done.')
    else:
        for _, row in abs_df.iterrows():
            face_viz(
                row,
                images_dir=images_dir,
                out_images_dir=out_images_dir,
                force=force,
                draw_inferred_box=draw_inferred_box,
            )


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('images_dir', '', '')
    flags.DEFINE_string('out_images_dir', '', '')
    flags.DEFINE_string('face_landmarks_file', '', '')
    flags.DEFINE_integer('max_images', 0, '')
    flags.DEFINE_boolean('parallel_compute', True, '')
    flags.DEFINE_boolean('force', False, '')
    flags.DEFINE_boolean('draw_inferred_box', False, '')

    flags.DEFINE_integer(
        'num_cpus', -1,
        'Number of CPUs to use in parallel. If less than 1 (including the default value -1), use all CPUs.'
    )

    app.run(main)
