import collections
import os
import glob

import cv2
from guided_filter_pytorch.guided_filter import GuidedFilter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from mydataset import MyDataset
from net import MaskGenerator, ResiduePredictor


def infer_palette(img_path, num_colors):
    cache_fn = img_path + f'.palette_{num_colors}_cache.npy'

    if os.path.exists(cache_fn):
        center = np.load(cache_fn)
    else:
        img_path = str(img_path)
        img = cv2.imread(img_path)[:, :, [2, 1, 0]]
        size = img.shape[:2]
        vec_img = img.reshape(-1, 3)
        model = KMeans(n_clusters=num_colors)
        pred = model.fit_predict(vec_img)
        pred_img = np.tile(pred.reshape(*size, 1), (1, 1, 3))
        center = model.cluster_centers_
        center = np.copy(center)
        np.save(cache_fn, center)

    return center, None


InferredBlob = collections.namedtuple('InferredBlob', [
    'primary_color_layers',
    'pred_unmixed_rgb_layers',
    'processed_alpha_layers',
    'reconst_img',
    'target_img',
])


def save_inferred_blob_as_image(inferred_blob, save_dir='dev_save'):
    os.makedirs(f'{save_dir}', exist_ok=True)
    for f in glob.glob(f'{save_dir}/*'):
        os.remove(f)

    primary_color_layers = inferred_blob.primary_color_layers
    pred_unmixed_rgb_layers = inferred_blob.pred_unmixed_rgb_layers
    processed_alpha_layers = inferred_blob.processed_alpha_layers
    reconst_img = inferred_blob.reconst_img
    target_img = inferred_blob.target_img

    save_image(primary_color_layers[0, :, :, :, :], f'{save_dir}/primary_color_layers.png')
    save_image(reconst_img[0, :, :, :], f'{save_dir}/reconst_img.png')
    save_image(target_img[0, :, :, :], f'{save_dir}/target_img.png')

    RGBA_layers = torch.cat((pred_unmixed_rgb_layers, processed_alpha_layers), dim=2)  # out: bn, ln, 4, h, w
    for i in range(RGBA_layers.shape[1]):
        save_image(RGBA_layers[0, i, :, :, :], f'{save_dir}/img-layer-{i}.png')
    mono_RGBA_layers = torch.cat((primary_color_layers, processed_alpha_layers), dim=2)  # out: bn, ln, 4, h, w
    for i in range(mono_RGBA_layers.shape[1]):
        save_image(mono_RGBA_layers[0, i, :, :, :], f'{save_dir}/mono-img-layer-{i}.png')
    mono_RGBA_layers_composed = (primary_color_layers * processed_alpha_layers).sum(dim=1)[0, :, :, :]
    save_image(mono_RGBA_layers_composed, f'{save_dir}/mono-img-reconst_img.png')


class Infer(object):
    def __init__(self, manual_colors, resources_dir=f'{os.path.dirname(__file__)}/resources'):
        self.manual_colors = manual_colors
        self.num_primary_color = num_primary_color = len(manual_colors)
        assert num_primary_color == 7  # model dependent.
        self.device = device = 'cuda'

        mask_generator = MaskGenerator(num_primary_color).to(device)
        residue_predictor = ResiduePredictor(num_primary_color).to(device)

        path_mask_generator = f'{resources_dir}/mask_generator.pth'
        path_residue_predictor = f'{resources_dir}/residue_predictor.pth'

        mask_generator.load_state_dict(torch.load(path_mask_generator))
        residue_predictor.load_state_dict(torch.load(path_residue_predictor))

        mask_generator.eval()
        residue_predictor.eval()

        self.mask_generator = mask_generator
        self.residue_predictor = residue_predictor

        self.csv_path = f'{resources_dir}/sample.csv'

    def infer(self, img_path):
        device = self.device
        manual_colors = self.manual_colors
        num_primary_color = self.num_primary_color
        mask_generator = self.mask_generator
        residue_predictor = self.residue_predictor

        test_dataset = MyDataset(self.csv_path, num_primary_color, mode='test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        test_dataset.imgs_path[0] = img_path
        test_dataset.palette_list[0] = manual_colors

        with torch.no_grad():
            target_img, primary_color_layers = next(iter(test_loader))
            target_img = cut_edge(target_img)
            target_img = target_img.to(device)  # bn, 3ch, h, w
            primary_color_layers, primary_color_pack = self.replace_color_pipleline(primary_color_layers, manual_colors)

            pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
            pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
            processed_alpha_layers = alpha_normalize(pred_alpha_layers)
            processed_alpha_layers = proc_guidedfilter(processed_alpha_layers, target_img)  # Option
            processed_alpha_layers = alpha_normalize(processed_alpha_layers)  # Option
            mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2)  #shape: bn, ln, 4, h, w
            mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1, target_img.size(2), target_img.size(3))
            residue_pack = residue_predictor(target_img, mono_color_layers_pack)
            residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
            pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)
            reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)

        return InferredBlob(
            primary_color_layers=primary_color_layers,
            pred_unmixed_rgb_layers=pred_unmixed_rgb_layers,
            processed_alpha_layers=processed_alpha_layers,
            reconst_img=reconst_img,
            target_img=target_img,
        )

    def replace_color_pipleline(self, primary_color_layers, manual_colors):
        device = self.device

        primary_color_layers = primary_color_layers.to(device)
        primary_color_layers = replace_color(primary_color_layers, manual_colors)  #ここ
        primary_color_pack = primary_color_layers.view(primary_color_layers.size(0), -1, primary_color_layers.size(3),
                                                       primary_color_layers.size(4))
        primary_color_pack = cut_edge(primary_color_pack)
        primary_color_layers = primary_color_pack.view(primary_color_pack.size(0), -1, 3, primary_color_pack.size(2),
                                                       primary_color_pack.size(3))

        return primary_color_layers, primary_color_pack


# Utility functions blow.


def replace_color(primary_color_layers, manual_colors):
    temp_primary_color_layers = primary_color_layers.clone()
    for layer in range(len(manual_colors)):
        for color in range(3):
            temp_primary_color_layers[:, layer, color, :, :].fill_(manual_colors[layer][color])
    return temp_primary_color_layers


def cut_edge(target_img):
    resize_scale_factor = 1  # default
    target_img = F.interpolate(target_img, scale_factor=resize_scale_factor, mode='area')
    h = target_img.size(2)
    w = target_img.size(3)
    h = h - (h % 8)
    w = w - (w % 8)
    target_img = target_img[:, :, :h, :w]
    return target_img


def alpha_normalize(alpha_layers):
    return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)


def proc_guidedfilter(alpha_layers, guide_img):
    guide_img = (guide_img[:, 0, :, :] * 0.299 + guide_img[:, 1, :, :] * 0.587 + guide_img[:, 2, :, :] * 0.114).unsqueeze(1)

    for i in range(alpha_layers.size(1)):
        # layerは，bn, 1, h, w
        layer = alpha_layers[:, i, :, :, :]

        processed_layer = GuidedFilter(3, 1 * 1e-6)(guide_img, layer)
        if i == 0:
            processed_alpha_layers = processed_layer.unsqueeze(1)
        else:
            processed_alpha_layers = torch.cat((processed_alpha_layers, processed_layer.unsqueeze(1)), dim=1)

    return processed_alpha_layers