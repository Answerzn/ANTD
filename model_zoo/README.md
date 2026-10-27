# Checkpoints

Place both the task-specific ANT weights and the shared diffusion weights here. Filenames are case-sensitive.

**Download:** [ANTD on Baidu Netdisk](https://pan.baidu.com/s/179xOEEyei7qjDZyawkMBfw?pwd=antd) · **Access code: `antd`**. The shared folder also contains the SIDD RGB dataset and test datasets; see the [data preparation guide](../README.md#data-preparation) for their locations.

| File | Size (bytes) | Source/status |
|:--|--:|:--|
| `ANT_blind.pth` | 22,638,861 | ANTD authors; shared download above |
| `ANT_sidd.pth` | 22,638,861 | ANTD authors; shared download above |
| `256x256_diffusion_uncond.pt` | 2,211,383,297 | [guided-diffusion](https://github.com/openai/guided-diffusion), ImageNet 256×256 unconditional |

The hashes below identify the files present when preparing this release. They do not identify the training iteration or independently establish model provenance.

```text
05c6d056f1c22b9399d1600c835f4012998dbb080571971d7b26fa83a5fea344  ANT_blind.pth
110650df52475d1da2cbfc5ba0358fee0b158969adfd480a1fb06ec2cac89091  ANT_sidd.pth
a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1  256x256_diffusion_uncond.pt
```

On Linux, use `sha256sum model_zoo/*.pth model_zoo/*.pt`; in PowerShell, use `Get-FileHash model_zoo/*.pth,model_zoo/*.pt -Algorithm SHA256`.

ANT weights are plain PyTorch state dictionaries. Training saves the same format as `<iteration>_G.pth`. Both evaluation scripts expose `--model_zoo`, `--model_name` (stem without extension) and `--diff_path`. Use weights from trusted sources; checkpoints are loaded with PyTorch.

Do not substitute a conditional or different-resolution diffusion checkpoint. The architecture must match `models/network_diffusion_model.py` exactly. Large weight files are excluded from Git and must be downloaded separately using the link above.
