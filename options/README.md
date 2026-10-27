# Training configuration reference

Both files use UTF-8 JSON with `//` comments, parsed by `utils/utils_option.py`. Run commands from the repository root. Relative paths are resolved from the working directory, not from the option file. Use local paths with forward slashes, e.g. `D:/datasets/SIDD_medium/gt`. The simple comment parser does not support URLs or paths containing `//` inside string values.

| Task | File |
|:--|:--|
| Synthetic blind RGB denoising | [train_ant_blind_denoising.json](train_ant_blind_denoising.json) |
| Paired SIDD RGB denoising | [train_ant_sidd_HL.json](train_ant_sidd_HL.json) |

## Change before running

| Key | Default / meaning | When to change |
|:--|:--|:--|
| `task` | Task-specific experiment name | Use a new name for every independent run |
| `gpu_ids` | `[0]`, physical CUDA IDs | Set to available GPUs; paper setup uses two |
| `path.root` | `experiments` | Output storage location |
| `datasets.train.dataroot_H` | Clean RGB folder | Required: point to your training data |
| `datasets.train.dataroot_L` | SIDD noisy folder; null for Gaussian | Required for paired SIDD |
| `datasets.test.dataroot_H/L` | Validation roots | Point to the correct held-out split |
| `pre_trained.pre_trained_diff_path` | `model_zoo/256x256_diffusion_uncond.pt` | Required diffusion checkpoint |
| `datasets.train.dataloader_batch_size` | 2 total samples per optimizer step | Paper: 4 across 2 GPUs; lower if memory is insufficient |
| `datasets.train.dataloader_num_workers` | 0 total workers | Portable Windows default; Linux can try 4-8 |
| `train.total_iter` | 600000 | Finite release default, not specified by the paper |

## Data and validation

| Key | Value / behavior |
|:--|:--|
| `model` | `antd`; uses `models/model_antd.py` |
| `scale` / `n_channels` | 1 / 3; provided models perform RGB denoising |
| `datasets.*.name` | Human-readable logging label |
| `datasets.*.dataset_type` | `antd_blind_denoising` or `antd_sidd`; both are registered loaders |
| `datasets.train.dataloader_shuffle` | true; shuffles training samples |
| `datasets.train.H_size` | 256, Gaussian loader's first crop; must be >= `lq_patchsize` |
| `datasets.train.lq_patchsize` | 256, final crop; use multiples of 64 |
| `datasets.train.sigma_range` | [5, 250], uniform Gaussian sigma in 8-bit units |
| `datasets.test.sigma` | 50, Gaussian validation noise level |
| `datasets.test.noise_seed` | 0, reset per validation image; unaffected by training RNG |
| `datasets.test.lq_patchsize` | Retained for loader configuration; no validation crop is performed |
| `train.validation_border` | 0, RGB PSNR without border shaving |

The SIDD loader matches noisy/clean identifiers. It requires equal dimensions and training images at least as large as the crop. Gaussian training upsizes undersized images before cropping. Both validation loaders use full images, with batch size 1.

## Network: keep fixed for released checkpoints

| `netG` key | Value | Meaning |
|:--|:--|:--|
| `net_type` | `noise_t_predictor` | Existing registry identifier for ANT |
| `in_nc` | 3 | RGB channels |
| `nc_config` | [2, 2, 2, 2, 2, 2, 2] | Compensation branch block counts |
| `t_config` | [2, 2, 2, 1] | Timestep branch block counts |
| `dim` | 32 | Base feature width |
| `drop_path_rate` | 0.0 | Stochastic depth rate |
| `input_resolution` | 128 | Attention construction hint; does not set the crop size |
| `init_type` | orthogonal | Generic initialization for training from scratch |
| `init_bn_type` | uniform | BatchNorm initialization policy |
| `init_gain` | 0.2 | Initialization gain |

ZeroConv is initialized to zero by the network initialization logic. Strict checkpoint loading replaces initialized weights. The architecture must also match the constants in both evaluation entry points if you train a different model.

The frozen diffusion backbone uses the defaults in `models/network_diffusion_model.py`. The schedule in `utils/utils_sampling.py` uses 1000 timesteps and linear β from 0.0001 to 0.02. Do not change these for the supplied weights.

## Optimization and output

| Key | Default / behavior |
|:--|:--|
| `path.pretrained_netG` | null; optional ANT initialization checkpoint for fine-tuning |
| `path.pretrained_optimizerG` | null; optional compatible optimizer state |
| `train.auto_resume` | true; latest numbered G checkpoint in this experiment takes precedence over initialization |
| `train.G_optimizer_reuse` | true; save/load Adam state; resume requires the matching iteration |
| `train.G_param_strict` | true; reject architecture/key mismatches |
| `train.manual_seed` | 0; seeds Python, NumPy and PyTorch |
| `train.G_lossfn_type/weight` | l2 / 1.0; reconstruction MSE |
| `train.t_lossfn_type/weight` | l2 / 1.0; normalized timestep MSE; paper λ=1 |
| `train.G_optimizer_type` | adam; the implementation constructs Adam |
| `train.G_optimizer_lr` | 1e-4; Adam uses default betas (0.9, 0.999), weight decay 0 |
| `train.G_scheduler_type` | MultiStepLR |
| `train.G_scheduler_milestones` | [100000, 200000, 300000, 400000, 500000], optimizer iterations |
| `train.G_scheduler_gamma` | 0.5; multiply LR at each milestone |
| `train.checkpoint_test` | 20000; validation interval in iterations |
| `train.checkpoint_save` | 20000; ANT/optimizer save interval |
| `train.checkpoint_print` | 200; loss logging interval |

All intervals must be positive. The stopping budget is configurable, and the final iteration is saved even if it is not a save interval. The supplied model does not use a discriminator or EMA, so stale `netD`, GAN, clipping and regularizer placeholders have been removed from the public configs.

For a fresh run, select a new `task`. For fine-tuning, use a new `task`, `auto_resume=false`, `pretrained_netG=<path>` and `pretrained_optimizerG=null`. For continued training, retain the task and keep its matching `*_G.pth` / `*_optimizerG.pth` pair.

See [the Chinese guide](../docs/USAGE_zh-CN.md) for practical examples and common errors.
