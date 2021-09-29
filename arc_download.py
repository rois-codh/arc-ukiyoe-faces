#!/usr/bin/env python3

import io
import os
import shutil
import urllib
import urllib.request

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import pandas as pd
import ray


@ray.remote
def download_and_check_image(url, filepath, force=False):
    try:
        if force or not os.path.isfile(filepath):
            # Load into bytes
            img_data = urllib.request.urlopen(url).read()

            # Convert to RGB if necessary
            img = Image.open(io.BytesIO(img_data))
            if img.mode != 'RGB':
                print('Grayscale -> RGB : %s' % filepath)
                arr = np.asarray(img)
                assert len(arr.shape) == 2
                arr = np.stack([arr] * 3, axis=-1)
                rgb_img = Image.fromarray(arr, mode='RGB')
                rgb_img.save(filepath)
            else:
                open(filepath, 'wb').write(img_data)
            img.close()
        return {'success': True}
    except Exception as e:
        err_msg = f'Download failed with {e} for image {os.path.basename(filepath)} from {url}'
        return {'success': False, 'err_msg': err_msg}


def main(_):
    FLAGS = flags.FLAGS
    resource_dir = './resource'
    metadata_file = f'{resource_dir}/arc_metadata.csv'
    images_dir = os.path.join(FLAGS.dir, 'arc_images')
    log_file = os.path.join(FLAGS.dir, 'arc_images_download.log')
    force = FLAGS.force

    os.makedirs(FLAGS.dir, exist_ok=True)
    for fn in ['arc_face.csv', 'arc_metadata.csv']:
        shutil.copyfile(f'{resource_dir}/{fn}', f'{FLAGS.dir}/{fn}')

    ray.shutdown()
    ray.init()

    metadata = pd.read_csv(metadata_file)
    os.makedirs(images_dir, exist_ok=True)
    futures = []
    for _, row in metadata.iterrows():
        url = row['LargeImageURL']
        filepath = os.path.join(images_dir, row.filename)
        futures.append(download_and_check_image.remote(url, filepath, force))
    logging.info(f'Paralleling downloading: {len(futures)} tasks running.')
    results = ray.get(futures)
    logging.info(f'Paralleling downloading: {len(futures)} tasks done.')

    with open(log_file, 'w') as fout:
        for result in results:
            if not result['success']:
                err_msg = result['err_msg']
                print(err_msg, file=fout)
    logging.info('Check %s for logs' % log_file)
    ray.shutdown()


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    flags.DEFINE_string('dir', f'{script_dir}/scratch',
                        'Directory in which the downloaded dataset is stored')
    flags.DEFINE_boolean('force', False,
                         'Force redownloading of already downloaded images')

    app.run(main)
