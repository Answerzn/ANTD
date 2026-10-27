from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
import numpy as np  

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss, GANLoss
from models.loss_ssim import SSIMLoss

from models.network_diffusion_model import UNetModel


from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip

import utils.utils_sampling as utils_sampling

import torch.nn.functional as F


class Model_ANTD (ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(Model_ANTD , self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option

        self.noise_t_predictor = define_G(opt)  # ANT Branch
        self.noise_t_predictor = self.model_to_device(self.noise_t_predictor)

        # Freeze diffusion model parameters: we only train noise_t_pred
        self.diff_model = UNetModel()
        # A frozen backbone still propagates gradients to ANT through its input.
        # Keep it local under DDP: it has no parameters to synchronize.
        self.diff_model = self.diff_model.to(self.device)
        if not self.opt['dist'] and self.opt['num_gpu'] > 1:
            self.diff_model = nn.DataParallel(self.diff_model)
        for p in self.diff_model.parameters():
            p.requires_grad = False
        self.diff_model.eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.noise_t_predictor.train()              # set train mode
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_noise_t_predictor = self.opt['path']['pretrained_netG']
        if load_path_noise_t_predictor is not None:
            print('Loading model for noise_t_predictor [{:s}] ...'.format(load_path_noise_t_predictor))
            self.load_network(load_path_noise_t_predictor, self.noise_t_predictor, strict=self.opt_train['G_param_strict'], param_key='params')

        load_path_diff = self.opt['pre_trained']['pre_trained_diff_path']
        print('Loading model for load_path_diff [{:s}] ...'.format(load_path_diff))
        self.load_network(load_path_diff, self.diff_model, strict=True, param_key='params_diffusion')


    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)


    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.noise_t_predictor, 'G', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)


    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        # ------------------------------------
        # 1) G_loss
        # ------------------------------------
        if self.opt_train['G_lossfn_weight'] > 0:
            G_lossfn_type = self.opt_train['G_lossfn_type']
            if G_lossfn_type == 'l1':
                self.G_lossfn = nn.L1Loss().to(self.device)
            elif G_lossfn_type == 'l2':
                self.G_lossfn = nn.MSELoss().to(self.device)
            elif G_lossfn_type == 'l2sum':
                self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
            elif G_lossfn_type == 'ssim':
                self.G_lossfn = SSIMLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
            self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        else:
            print('Do not use pixel loss.')
            self.G_lossfn = None

        # ------------------------------------
        # 2) t_loss
        # ------------------------------------
        t_lossfn_type = self.opt_train['t_lossfn_type']
        if t_lossfn_type == 'l1':
            self.t_lossfn = nn.L1Loss().to(self.device)
        elif t_lossfn_type == 'l2':
            self.t_lossfn = nn.MSELoss().to(self.device)
        self.t_lossfn_weight = self.opt_train['t_lossfn_weight']


    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        # Only optimize parameters of noise_t_pred
        G_optim_params = []
        for k, v in self.noise_t_predictor.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)


        # count trainable parameters
        def count_trainable_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # print a compact summary
        print('diff_model trainable params:', count_trainable_params(self.diff_model))
        print('noise_t_predictor trainable params:', count_trainable_params(self.noise_t_predictor))
        print('total trainable params (model):',
              count_trainable_params(self.noise_t_predictor) + count_trainable_params(self.diff_model))



    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'fixed':
            print("Using fixed learning rate, no scheduler is applied.")
            
        else:
            raise NotImplementedError


    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)
        self.tt = data['tt'].unsqueeze(-1).to(self.device)
        self.L_paths = data['L_path']
        self.H_paths = data['H_path']

    def data_transform(self, X):
        """Map [0,1] -> [-1,1]."""
        return 2 * X - 1.0

    def data_inverse_transform(self, X):
        """Inverse of data_transform: map [-1,1] -> [0,1]."""
        return (X + 1.0) / 2.0

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):

        h, w = self.L.size()[-2:]
        paddingBottom = int(np.ceil(h/64)*64-h)
        paddingRight = int(np.ceil(w/64)*64-w)
        self.L = F.pad(self.L, (0, paddingRight, 0, paddingBottom), mode='reflect')  # left, right, up, down

        # data transform
        self.L = self.data_transform(self.L)
        self.x_s, self.t_pred = self.noise_t_predictor(self.L)

        # get the diffusion start point: x_start
        x_start, t_cont, alpha_t = utils_sampling.time_step_schedule(x0=self.x_s, t_pred=self.t_pred, T=1000, beta=None)

        # predicted timestep
        t_for_model = t_cont

        et = self.diff_model(x_start, t_for_model)

        if et.size(1) == 6:
            et = et[:, :3]

        sqrt_alpha = alpha_t.sqrt().unsqueeze(-1)      # shape broadcasting to (N,1,1,1)
        sqrt_one_minus_alpha = (1 - alpha_t).sqrt().unsqueeze(-1)

        # just one step inference
        x0_pred = (x_start - sqrt_one_minus_alpha * et) / sqrt_alpha

        x0_pred = x0_pred[..., :h, :w]

        # data inverse transform
        self.E = self.data_inverse_transform(x0_pred)


    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        # If GAN training is enabled and a discriminator exists, follow GAN update schedule
        if self.is_train and hasattr(self, 'netD') and (self.opt_train.get('D_lossfn_weight', 0) > 0 or self.D_lossfn_weight > 0):
            # ----------------------
            # Update G (generator / predictor)
            # ----------------------
            for p in self.netD.parameters():
                p.requires_grad = False

            self.G_optimizer.zero_grad()
            self.netG_forward()
            loss_G_total = 0

            # pixel / reconstruction loss
            if self.G_lossfn_weight > 0:
                G_loss = self.G_lossfn(self.E, self.H)
                loss_G_total += self.G_lossfn_weight * G_loss

            # time loss
            if self.t_lossfn_weight > 0:
                time_loss = self.t_lossfn(self.t_pred, self.tt)
                loss_G_total += self.t_lossfn_weight * time_loss

            # Backprop G
            loss_G_total.backward()
            # clip grads
            self.G_optimizer.step()

            # logging
            if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:
                if self.G_lossfn_weight > 0:
                    self.log_dict['G_loss'] = G_loss.item()


        else:
            # fallback to original non-GAN update
            self.G_optimizer.zero_grad()
            self.netG_forward()
            G_loss = self.G_lossfn(self.E, self.H)
            loss = self.G_lossfn_weight * G_loss
            # time loss
            if self.t_lossfn_weight > 0:
                time_loss = self.t_lossfn(self.t_pred, self.tt)
                loss += self.t_lossfn_weight * time_loss
            loss.backward()
            self.G_optimizer.step()
            self.log_dict['G_loss'] = G_loss.item()
            if self.t_lossfn_weight > 0:
                self.log_dict['time_loss'] = time_loss.item()

        if self.opt_train.get('E_decay', 0) > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.noise_t_predictor.eval()
        # Rank-zero validation must not enter DDP forward collectives alone.
        predictor = self.noise_t_predictor
        self.noise_t_predictor = self.get_bare_model(predictor)
        with torch.no_grad():
            try:
                self.netG_forward()
            finally:
                self.noise_t_predictor = predictor
        self.noise_t_predictor.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.noise_t_predictor)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.noise_t_predictor)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.noise_t_predictor)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.noise_t_predictor)
        return msg
