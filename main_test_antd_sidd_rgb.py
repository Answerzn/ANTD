import os
import logging
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import lpips

from utils import utils_logger
from utils import utils_image as util
from utils.utils_antd import paired_image_paths, validate_image_size


def data_transform(X):
    return 2 * X - 1.0


def data_inverse_transform(X):
    return (X + 1.0) / 2.0


def main():

    # ----------------------------------------
    # Arguments
    # ----------------------------------------
    # EDIT via CLI: checkpoint paths, aligned PNG directories, device and results.
    # SIDD validation uses RGB PNG pairs, not RAW images or hidden benchmark GT.
    parser = argparse.ArgumentParser(description='Evaluate aligned SIDD RGB pairs (PSNR/SSIM/LPIPS).')
    parser.add_argument('--model_name', type=str, default='ANT_sidd', help='ANT checkpoint stem without .pth; case-sensitive on Linux.')
    parser.add_argument('--model_zoo', type=str, default='model_zoo', help='Directory containing the ANT checkpoint.')
    parser.add_argument('--diff_path', type=str, default='model_zoo/256x256_diffusion_uncond.pt', help='Frozen ImageNet diffusion checkpoint.')

    parser.add_argument('--L_path', type=str, default='testsets/sidd_validation/noisy',
                        help='Folder of noisy images')
    parser.add_argument('--H_path', type=str, default='testsets/sidd_validation/gt',
                        help='Folder of GT images')

    parser.add_argument('--results', type=str, default='results', help='Output root; images/log go to sidd_<model_name>.')
    parser.add_argument('--show_img', action='store_true', help='Display restored/GT images; omit on servers.')
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, cuda, or cuda:0.')

    args = parser.parse_args()

    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    for checkpoint in (os.path.join(args.model_zoo, args.model_name + '.pth'), args.diff_path):
        if not os.path.isfile(checkpoint):
            parser.error('Checkpoint not found: ' + checkpoint)
    L_paths, H_paths = paired_image_paths(args.L_path, args.H_path)
    n_channels = 3
    border = 0

    # ----------------------------------------
    # Logger
    # ----------------------------------------
    result_name = 'sidd_' + args.model_name
    E_path = os.path.join(args.results, result_name)
    util.mkdir(E_path)

    logger_name = 'Test_Denoising'
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, 'test.log'))
    logger = logging.getLogger(logger_name)

    logger.info(f'L_path: {args.L_path}')
    logger.info(f'H_path: {args.H_path}')
    logger.info(f'Model: {args.model_name}; diffusion: {args.diff_path}; device: {device}; pairs: {len(L_paths)}')

    # ----------------------------------------
    # LPIPS
    # ----------------------------------------
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # ----------------------------------------
    # Load model
    # ----------------------------------------
    from models.network_ant import Noise_T_Predictor
    from models.network_diffusion_model import UNetModel
    import utils.utils_sampling as utils_sampling

    # CHECKPOINT CONTRACT: keep identical to netG in the training options.
    noise_t_predictor = Noise_T_Predictor(
        in_nc=3, out_nc=3, 
        nc_config=[2,2,2,2,2,2,2],
        t_config=[2, 2, 2, 1],
        dim=32,
        input_resolution=128
    )
    noise_t_predictor.load_state_dict(
        torch.load(os.path.join(args.model_zoo, args.model_name + '.pth'),
                   map_location=device),
        strict=True
    )
    noise_t_predictor.eval().to(device)

    diff_model = UNetModel()
    diff_model.load_state_dict(torch.load(args.diff_path, map_location=device))
    diff_model.eval().to(device)

    # ----------------------------------------
    # Load image paths
    # ----------------------------------------
    # Pairs were validated by identifiers above; equal counts alone are insufficient.

    test_results = OrderedDict(psnr=[], ssim=[], lpips=[])

    # ----------------------------------------
    # Inference + Evaluation
    # ----------------------------------------
    for idx, (L_img, H_img) in enumerate(zip(L_paths, H_paths)):

        img_name, ext = os.path.splitext(os.path.basename(L_img))

        # img_name = os.path.basename(L_img)

        # ---- read noisy image ----
        img_L = util.imread_uint(L_img, n_channels=n_channels)
        validate_image_size(*img_L.shape[:2])
        img_L = util.uint2single(img_L)
        img_L = util.single2tensor4(img_L).to(device)

        with torch.no_grad():

            h, w = img_L.shape[-2:]
            pad_h = int(np.ceil(h / 64) * 64 - h)
            pad_w = int(np.ceil(w / 64) * 64 - w)
            img_L = F.pad(img_L, (0, pad_w, 0, pad_h), mode='reflect')

            img_L = data_transform(img_L)

            xs, t_pred = noise_t_predictor(img_L)

            x_t, t_cont, alpha_t = utils_sampling.time_step_schedule(
                x0=xs, t_pred=t_pred, T=1000, beta=None
            )


            et = diff_model(x_t, t_cont)
            if et.size(1) == 6:
                et = et[:, :3]

            sqrt_alpha = alpha_t.sqrt().unsqueeze(-1)
            sqrt_1m_alpha = (1 - alpha_t).sqrt().unsqueeze(-1)

            x0_pred = (x_t - sqrt_1m_alpha * et) / sqrt_alpha
            x0_pred = x0_pred[:, :, :h, :w]

            img_E = data_inverse_transform(x0_pred).clamp(0, 1)

        img_E = util.tensor2uint(img_E)

        # ---- read GT ----
        img_H = util.imread_uint(H_img, n_channels=n_channels)
        if img_E.shape != img_H.shape:
            raise ValueError('Noisy/GT image dimensions differ: {} / {}'.format(L_img, H_img))

        # ---- metrics ----
        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)

        # Ensure tensors are float32 (LPIPS expects float32 inputs matching model weights)
        imgE_t = torch.from_numpy((img_E / 255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
        imgH_t = torch.from_numpy((img_H / 255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

        # map to [-1, 1]
        imgE_t = imgE_t * 2 - 1
        imgH_t = imgH_t * 2 - 1


        with torch.no_grad():
            lpips_val = lpips_fn(imgE_t, imgH_t).item()

        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        test_results['lpips'].append(lpips_val)

        logger.info(
            f'{img_name} | PSNR {psnr:.2f} | SSIM {ssim:.4f} | LPIPS {lpips_val:.4f}'
        )

        if args.show_img:
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Restored / Ground truth')
        output_path = os.path.join(E_path, os.path.splitext(os.path.relpath(L_img, args.L_path))[0] + '.png')
        util.mkdir(os.path.dirname(output_path))
        util.imsave(img_E, output_path)

    # ----------------------------------------
    # Average
    # ----------------------------------------
    logger.info('============== Average Results ==============')
    logger.info(
        f'PSNR: {np.mean(test_results["psnr"]):.2f}, '
        f'SSIM: {np.mean(test_results["ssim"]):.4f}, '
        f'LPIPS: {np.mean(test_results["lpips"]):.4f}'
    )


if __name__ == '__main__':
    main()
