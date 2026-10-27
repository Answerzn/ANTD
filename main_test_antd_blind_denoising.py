import os.path
import logging
import argparse

import numpy as np
from collections import OrderedDict

import torch
from utils import utils_logger
from utils import utils_image as util
from utils.utils_antd import validate_image_size

import torch.nn.functional as F

import lpips


def data_transform(X):
    """Map [0,1] -> [-1,1]."""
    return 2 * X - 1.0

def data_inverse_transform(X):
    """Inverse of data_transform: map [-1,1] -> [0,1]."""
    return (X + 1.0) / 2.0


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    # EDIT via CLI: dataset, noise level, model paths, device, output directory.
    # This benchmark reads CLEAN RGB images and adds AWGN; sigma is not fed to ANT.
    parser = argparse.ArgumentParser(description='Evaluate blind RGB Gaussian denoising (PSNR/SSIM/LPIPS).')
    parser.add_argument('--model_name', type=str, default='ANT_blind', help='Checkpoint stem, without .pth; case-sensitive on Linux.')
    parser.add_argument('--testset_name', type=str, default='cbsd68', help='Clean RGB folder under --testsets: cbsd68, kodak24, McMaster, Urban100.')
    parser.add_argument('--noise_level_img', type=float, default=50, help='AWGN sigma in 0-255 units. Paper: 15, 25, 50, 100, 150, 200.')
    parser.add_argument('--show_img', action='store_true', help='Display images; omit on headless servers.')
    parser.add_argument('--model_zoo', type=str, default='model_zoo', help='path of model_zoo')
    parser.add_argument('--testsets', type=str, default='testsets', help='path of testing folder')
    parser.add_argument('--results', type=str, default='results/antd_denoising', help='path of results')
    parser.add_argument('--diff_path', type=str, default='model_zoo/256x256_diffusion_uncond.pt', help='path of diffusion unet')
    parser.add_argument('--noise_seed', type=int, default=0, help='Reset NumPy noise seed per image; keep 0 for the reference protocol.')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:0. Auto selects CUDA when available.')


    args = parser.parse_args()
    if args.noise_level_img < 0 or not np.isfinite(args.noise_level_img):
        parser.error('--noise_level_img must be finite and non-negative.')
    model_path = os.path.join(args.model_zoo, args.model_name + '.pth')
    for checkpoint in (model_path, args.diff_path):
        if not os.path.isfile(checkpoint):
            parser.error('Checkpoint not found: ' + checkpoint)

    n_channels = 3        # fixed, 1 for grayscale image, 3 for color image

    result_name = args.testset_name + '_' + args.model_name + '_' + format(args.noise_level_img, 'g')
    border = 0            # shave boader to calculate PSNR and SSIM

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = os.path.join(args.testsets, args.testset_name) # L_path, for Low-quality images
    H_path = L_path                               # H_path, for High-quality images
    E_path = os.path.join(args.results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)

    lpips_fn = lpips.LPIPS(net='alex').eval().to(device)

    # ----------------------------------------
    # load model
    # ----------------------------------------
    from models.network_ant import Noise_T_Predictor as net
    from models.network_diffusion_model import UNetModel
    import utils.utils_sampling as utils_sampling

    # ------- load noise_t_predictor -------
    model_path = os.path.join(args.model_zoo, args.model_name+'.pth')
    # CHECKPOINT CONTRACT: keep identical to netG in the training options.
    noise_t_predictor = net(in_nc=3, out_nc=3, nc_config=[2,2,2,2,2,2,2], t_config=[2,2,2,1], dim=32, drop_path_rate=0.0, input_resolution=128)
    noise_t_predictor.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    noise_t_predictor.eval().to(device)
    for k, v in noise_t_predictor.named_parameters():
        v.requires_grad = False
    # ------- load diffusion UNet -------
    diff_path = args.diff_path
    diff_model = UNetModel()
    ckpt = torch.load(diff_path, map_location=device)
    diff_model.load_state_dict(ckpt)
    diff_model.eval().to(device)
    for p in diff_model.parameters():
        p.requires_grad = False


    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), noise_t_predictor.parameters()))
    logger.info('Noise Time Predictor Params number: {:>16s} : {:<.4f} [M]'.format('#Params', number_parameters/10**6))

    logger.info('Diffusion UNet path: {:s}'.format(diff_path))
    number_parameters = sum(map(lambda x: x.numel(), diff_model.parameters()))
    logger.info('Pretrained Diffusion Params number: {:>16s} : {:<.4f} [M]'.format('#Params', number_parameters/10**6))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['lpips'] = []

    logger.info('model_name:{}, image sigma:{}, noise_seed:{}, device:{}'.format(args.model_name, args.noise_level_img, args.noise_seed, device))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None


    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        validate_image_size(*img_L.shape[:2])
        # Keep noisy inputs unclipped; clipping changes the evaluation protocol.
        rng = np.random.RandomState(args.noise_seed)
        img_L += rng.normal(0, args.noise_level_img / 255.0, img_L.shape)


        util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(args.noise_level_img)) if args.show_img else None


        
        img_L = util.single2tensor4(img_L)

        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E  ——  Diffusion Inversion Inference (single-step)
        # ------------------------------------
        with torch.no_grad():

            h, w = img_L.size()[-2:]
            paddingBottom = int(np.ceil(h/64)*64-h)
            paddingRight = int(np.ceil(w/64)*64-w)
            img_L = F.pad(img_L, (0, paddingRight, 0, paddingBottom), mode='reflect')  # left, right, up, down

            img_L = data_transform(img_L)  # [0,1] -> [-1,1]

            # ========== 1. 预测 noise 和 t ==========
            xs, t_pred = noise_t_predictor(img_L)

            # ========== 2. 构造 x_t ==========

            # noise = None
            # t_pred = torch.Tensor([[0.115]]).to(device)
            x_t, t_cont, alpha_t = utils_sampling.time_step_schedule(
                x0=xs,
                t_pred=t_pred,
                T=1000,
                beta=None
            )

            # print('predicted t:', t_cont)


            # ========== 4. 冻结扩散 UNet 预测 ε ==========
            et = diff_model(x_t, t_cont)


            if et.size(1) == 6:   # 防止 6 通道输出
                et = et[:, :3]

            # ========== 5. 反推 x0 ==========
            sqrt_alpha = alpha_t.sqrt().unsqueeze(-1)
            sqrt_one_minus_alpha = (1 - alpha_t).sqrt().unsqueeze(-1)

            x0_pred = (x_t - sqrt_one_minus_alpha * et) / sqrt_alpha

            x0_pred = x0_pred[:, :, :h, :w]

            # ========== 6. [-1,1] → [0,1] ==========
            img_E = data_inverse_transform(x0_pred)
            img_E = torch.clamp(img_E, 0, 1)

        img_E = util.tensor2uint(img_E)

        if need_H:


            # --------------------------------
            # (3) img_H
            # --------------------------------
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()

            # # patch
            # img_H = img_H[rnd_h_H:rnd_h_H + 128, rnd_w_H:rnd_w_H + 128, :]

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------
            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)

            imgE_t = torch.from_numpy(img_E.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            imgH_t = torch.from_numpy(img_H.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            imgE_t = imgE_t * 2.0 - 1.0
            imgH_t = imgH_t * 2.0 - 1.0
            with torch.no_grad():
                lpips_out = lpips_fn(imgE_t, imgH_t)
            lpips_val = float(lpips_out.detach().cpu().item())
            test_results['lpips'].append(lpips_val)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}; LPIPS: {:.4f}.'.format(img_name+ext, psnr, ssim, lpips_val))
            
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if args.show_img else None

        # ------------------------------------
        # save results
        # ------------------------------------
        # PNG avoids JPEG recompression before optional no-reference metrics.
        output_path = os.path.join(E_path, os.path.splitext(os.path.relpath(img, L_path))[0] + '.png')
        util.mkdir(os.path.dirname(output_path))
        util.imsave(img_E, output_path)

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
        logger.info('Average PSNR/SSIM/LPIPS(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}; LPIPS: {:.4f}'.format(result_name, ave_psnr, ave_ssim, ave_lpips))

if __name__ == '__main__':

    main()

