"""Aligned noisy/clean SIDD RGB pairs with corresponding random training crops."""

import random

import torch
from torch.utils.data import Dataset

from utils import utils_image as util
from utils import utils_sampling
from utils.utils_antd import paired_image_paths


class DatasetANTD(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.n_channels = opt['n_channels'] or 3
        self.l_patchsize = opt.get('lq_patchsize') or 256
        # EDIT in options: dataroot_L=noisy RGB; dataroot_H=aligned clean RGB.
        # Use identical relative names, or official SIDD NOISY/GT name pairs.
        self.paths_L, self.paths_H = paired_image_paths(opt['dataroot_L'], opt['dataroot_H'])
        betas = utils_sampling.get_beta_schedule(
            'linear', num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        self.sigma_ratio = utils_sampling.precompute_ratio(torch.from_numpy(betas).float())

    def __getitem__(self, index):
        L_path, H_path = self.paths_L[index], self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_L = util.imread_uint(L_path, self.n_channels)
        if img_H.shape != img_L.shape:
            raise ValueError('Noisy/GT dimensions differ: {} / {}'.format(L_path, H_path))
        if self.opt['phase'] == 'train':
            height, width = img_H.shape[:2]
            if min(height, width) < self.l_patchsize:
                raise ValueError('SIDD images must be at least lq_patchsize pixels: ' + H_path)
            top = random.randint(0, height - self.l_patchsize)
            left = random.randint(0, width - self.l_patchsize)
            img_H = img_H[top:top + self.l_patchsize, left:left + self.l_patchsize]
            img_L = img_L[top:top + self.l_patchsize, left:left + self.l_patchsize]

        img_H = util.single2tensor3(util.uint2single(img_H))
        img_L = util.single2tensor3(util.uint2single(img_L))
        sigma = torch.std(img_L - img_H)
        timestep = utils_sampling.find_best_t(self.sigma_ratio, sigma=2 * sigma) / 1000.0
        return {'L': img_L, 'H': img_H, 'tt': torch.tensor(timestep).float(),
                'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
