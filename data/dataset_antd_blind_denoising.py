"""Clean RGB images with on-the-fly AWGN and timestep supervision."""

import math
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import utils_image as util
from utils import utils_sampling


class DatasetANTD(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.n_channels = opt['n_channels'] or 3
        self.patch_size = opt.get('H_size') or 256
        self.l_patchsize = opt.get('lq_patchsize') or 256
        # EDIT in options: clean image directories, sigma range and crop sizes.
        # Validation uses fixed noise, matching the standalone benchmark script.
        self.sigma_range = opt.get('sigma_range', [5, 250])
        self.sigma_test = opt.get('sigma', 50)
        self.seed = opt.get('noise_seed', 0)
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        if not self.paths_H:
            raise ValueError('dataroot_H must contain clean RGB images.')
        if self.patch_size < self.l_patchsize:
            raise ValueError('H_size must be >= lq_patchsize.')
        if len(self.sigma_range) != 2 or not 0 <= self.sigma_range[0] <= self.sigma_range[1]:
            raise ValueError('sigma_range must be [minimum, maximum], in 0-255 units.')
        betas = utils_sampling.get_beta_schedule(
            'linear', num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        self.sigma_ratio = utils_sampling.precompute_ratio(torch.from_numpy(betas).float())

    def __getitem__(self, index):
        path = self.paths_H[index]
        img_H = util.imread_uint(path, self.n_channels)
        if self.opt['phase'] == 'train':
            height, width = img_H.shape[:2]
            if min(height, width) < self.patch_size:
                factor = self.patch_size / min(height, width)
                img_H = cv2.resize(img_H, (math.ceil(width * factor), math.ceil(height * factor)),
                                   interpolation=int(np.random.choice([cv2.INTER_CUBIC, cv2.INTER_AREA])))
            height, width = img_H.shape[:2]
            top = random.randint(0, height - self.patch_size)
            left = random.randint(0, width - self.patch_size)
            img_H = img_H[top:top + self.patch_size, left:left + self.patch_size]
            img_H = util.augment_img(img_H, mode=random.randint(0, 7))
            top = random.randint(0, self.patch_size - self.l_patchsize)
            left = random.randint(0, self.patch_size - self.l_patchsize)
            img_H = img_H[top:top + self.l_patchsize, left:left + self.l_patchsize]
            sigma = np.random.uniform(*self.sigma_range)
            rng = np.random
        else:
            sigma = self.sigma_test
            rng = np.random.RandomState(self.seed)

        img_H = util.uint2single(img_H)
        img_L = img_H.copy()
        img_L += rng.normal(0, sigma / 255.0, img_L.shape)
        # Do not clip AWGN inputs: clipping changes the benchmark distribution.
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        sigma = torch.std(img_L - img_H)
        timestep = utils_sampling.find_best_t(self.sigma_ratio, sigma=2 * sigma) / 1000.0
        return {'L': img_L, 'H': img_H, 'tt': torch.tensor(timestep).float(),
                'L_path': path, 'H_path': path}

    def __len__(self):
        return len(self.paths_H)
