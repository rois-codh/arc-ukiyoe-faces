#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd


def change_extension(filename_or_filepath, new_extension):
    if not new_extension.startswith('.'):
        raise ValueError(
            f'new_extension should shart with a dot, but `{new_extension}` is given.'
        )
    root, _ = os.path.splitext(filename_or_filepath)
    return f'{root}{new_extension}'


def ensure_dir_exists(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def pd_get_columns(df_or_series):
    '''Get the columns from a DataFrame or a Series (when it's viewed as a row of a DataFrame).'''
    if isinstance(df_or_series, pd.DataFrame):
        return df_or_series.columns
    elif isinstance(df_or_series, pd.Series):
        if not len(df_or_series.axes) == 1:
            raise ValueError('MultiIndex is not supported.')
        return df_or_series.axes[0]
    else:
        raise ValueError(
            f'Cannot get columns from a variable of type {type(df_or_series)}')


def get_landmark_xy_by_name(df_or_series, name, as_array=False):
    preifx = 'landmarks_'
    x = df_or_series[f'{preifx}{name}_x']
    y = df_or_series[f'{preifx}{name}_y']
    result = x, y
    if as_array:
        result = np.array(result)
    return result


def set_landmark_xy_by_name(df_or_series, name, xy):
    preifx = 'landmarks_'
    x, y = xy[0], xy[1]
    df_or_series[f'{preifx}{name}_x'] = x
    df_or_series[f'{preifx}{name}_y'] = y


def get_virtualpoint_xy_by_name(df_or_series, name, as_array=False):
    if name == 'top_left':
        prefix = 'bounding_box_'
        left = df_or_series[f'{prefix}left']
        top = df_or_series[f'{prefix}top']
        x, y = left, top
    elif name == 'bottom_right':
        prefix = 'bounding_box_'
        left = df_or_series[f'{prefix}left']
        width = df_or_series[f'{prefix}width']
        right = df_or_series[f'{prefix}left'] + width
        top = df_or_series[f'{prefix}top']
        height = df_or_series[f'{prefix}height']
        bottom = df_or_series[f'{prefix}top'] + height
        x, y = right, bottom
    else:
        raise ValueError(f'Unsupported name {name}')
    result = x, y
    if as_array:
        result = np.array(result)
    return result


def set_virtualpoint_xy_by_name(df_or_series, name, xy):
    x, y = xy[0], xy[1]
    if name == 'top_left':
        prefix = 'bounding_box_'
        left, top = x, y
        df_or_series[f'{prefix}left'] = left
        df_or_series[f'{prefix}top'] = top
    elif name == 'bottom_right':
        # WARNING: it works only with correct top_left
        prefix = 'bounding_box_'
        left = df_or_series[f'{prefix}left']
        right = x
        width = right - left
        df_or_series[f'{prefix}width'] = width
        top = df_or_series[f'{prefix}top']
        bottom = y
        height = bottom - top
        df_or_series[f'{prefix}height'] = height
    else:
        raise ValueError(f'Unsupported name {name}')


def get_landmarks_name(df_or_series):
    '''Get the landmark names, e.g. `landmarks_eye_left`, with out `_x` or `_y`.'''

    prefix = 'landmarks_'
    results = []
    for column in pd_get_columns(df_or_series):
        if column.startswith(prefix):
            assert column.endswith('_x') or column.endswith('_y')
            mark = column
            mark = mark[:-2]
            mark = mark[len(prefix):]
            if mark not in results:
                results.append(mark)
    return results


def get_bounding_box_left_right_top_bottom(df_or_series, as_array=False):
    prefix = 'bounding_box_'

    left = df_or_series[f'{prefix}left']
    width = df_or_series[f'{prefix}width']
    right = df_or_series[f'{prefix}left'] + width
    top = df_or_series[f'{prefix}top']
    height = df_or_series[f'{prefix}height']
    bottom = df_or_series[f'{prefix}top'] + height

    result = left, right, top, bottom
    if as_array:
        result = np.array(result)
    return result


def rel_df_to_abs_df(df):
    return convert_df_scale(df, target='abs')


def abs_df_to_rel_df(df):
    return convert_df_scale(df, target='rel')


def convert_df_scale(df, target='abs'):
    allowed_target = ['abs', 'rel']
    if target not in allowed_target:
        raise ValueError(
            f'target must be in {allowed_target}, but got {target}')
    new_df = pd.DataFrame()

    for column in pd_get_columns(df):
        if column.startswith('bounding_box') or column.startswith('landmarks'):
            if column.endswith('left') or column.endswith(
                    'width') or column.endswith('x'):
                ref_axis = 'width'
            elif column.endswith('top') or column.endswith(
                    'height') or column.endswith('y'):
                ref_axis = 'height'
            else:
                raise ValueError(
                    f'Cannot determine ref_axis for column {column}')

            if target == 'abs':
                new_df[column] = df[column] * df[ref_axis]
            elif target == 'rel':
                new_df[column] = df[column] / df[ref_axis]
        else:  # keep intact
            new_df[column] = df[column]

    return new_df


def dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return ((a - b)**2).sum()**(1 / 2)


def rotate(a, theta):
    # theta is in radians.
    a = np.array(a)
    matrix = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    a_prime = np.dot(matrix, a)
    return a_prime


def norm(a):
    a = np.array(a)
    return dist(a, [0., 0.])


def normalize(a):
    a = np.array(a)
    n = norm(a)
    if n < 1e-8:
        return np.array([0., 0.])
    else:
        return a / n


def radians_normalize(a):
    # normalize to (-pi, pi]
    while a <= -np.pi:
        a = a + np.pi * 2
    while a > np.pi:
        a = a - np.pi * 2
    return a


def radians_to_degree(a):
    return radians_normalize(a) / np.pi * 180.0


def compute_specs(abs_row):
    # Conclution:
    # - two eyes be hotizontal
    # - mean ['eye_left', 'eye_right']  to Y=480 (e.g 15/32)
    # - mean ['eye_left', 'eye_right']  to X=512 (e.g. 16/32)
    # - mean ['mouth_left', 'mouth_right'] to Y=768 (e.g 24/32)
    # now the frame is fixed.

    eye_left = get_landmark_xy_by_name(abs_row, 'eye_left', as_array=True)
    eye_right = get_landmark_xy_by_name(abs_row, 'eye_right', as_array=True)
    mouth_left = get_landmark_xy_by_name(abs_row, 'mouth_left', as_array=True)
    mouth_right = get_landmark_xy_by_name(abs_row,
                                          'mouth_right',
                                          as_array=True)

    eye_mean = (eye_left + eye_right) / 2.
    mouth_mean = (mouth_left + mouth_right) / 2.

    vec = (eye_right - eye_left)
    vec = rotate(vec, np.pi * 1 / 2)
    vec = normalize(vec)
    vec = vec * (norm(mouth_mean - eye_mean) * (1. * (512 - 480) /
                                                (768 - 480)))
    c = eye_mean + vec

    vec = normalize(vec)
    alpha = np.angle(vec[0] + 1j * vec[1])
    alpha = np.pi * 1 / 2 - alpha
    alpha = radians_normalize(alpha)

    l = norm(mouth_mean - eye_mean) / (1. * (768 - 480) / 1024)

    return l, c, alpha
