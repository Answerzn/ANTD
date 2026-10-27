# ANTD 使用指南与注意事项

对应论文：**Adapting Pre-trained Diffusion Model for Blind Image Denoising via Noise Compensation and Timestep Prediction，ICIP 2026**。

ANTD 包含可训练的 ANT 分支和冻结的扩散主干。训练只更新 ANT；推理需要同时加载 ANT 与扩散模型两个权重文件。英文主页见 [README](../README.md)，参数逐项说明见 [options/README.md](../options/README.md)。

## 1. 环境安装

所有命令均在项目根目录运行。推荐 Python 3.9；本次检查使用 PyTorch 2.4.1、torchvision 0.19.1 和 RTX 4090。训练需要 CUDA，多卡 DDP 推荐使用 Linux。

```bash
conda create -n antd python=3.9 -y
conda activate antd
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

根据驱动选择匹配的 PyTorch 安装源，torch 与 torchvision 必须配套。原来的完整机器环境导出含有本地 Conda 路径和不兼容的版本组合，已替换为项目所需依赖。可选的无参考评价依赖单独安装：

```bash
python -m pip install -r requirements-iqa.txt
```

LPIPS 和 PyIQA 首次运行可能下载评价模型。离线环境需要提前准备缓存。`--device auto` 自动选择 GPU 或 CPU；训练入口要求 GPU。CPU 可用于推理检查，但扩散主干较大，速度较慢。

## 2. 准备权重

**模型权重、SIDD RGB 数据集和测试数据集**统一放在名为 **ANTD** 的百度网盘分享中：

> **[百度网盘下载](https://pan.baidu.com/s/179xOEEyei7qjDZyawkMBfw?pwd=antd)**  
> **提取码：`antd`**

下载并解压后，将权重放入下方目录，数据按第 3 节的结构摆放；也可保存在其他位置，再修改配置或测试命令中的路径。

```text
model_zoo/
├── ANT_blind.pth
├── ANT_sidd.pth
└── 256x256_diffusion_uncond.pt
```

- `ANT_blind.pth`：合成高斯噪声模型。
- `ANT_sidd.pth`：SIDD 真实 RGB 噪声模型。注意 Linux 区分大小写。
- `256x256_diffusion_uncond.pt`：两项任务共用的冻结扩散主干，来自 [guided-diffusion](https://github.com/openai/guided-diffusion) 的 ImageNet 256×256 unconditional 模型。

已有本地权重可直接使用，SHA-256 和文件大小见 [权重说明](../model_zoo/README.md)。Git 忽略规则排除了大权重文件，请通过上面的网盘链接单独下载。

## 3. 数据组织

### 合成噪声训练

将 Waterloo Exploration、DIV2K、Flickr2K 的干净 RGB 图像放入 `datasets/clean_train/`，可以保留子目录。训练时在线采样 σ∈[5,250]，无需准备 noisy 文件夹。

```text
datasets/clean_train/
├── Waterloo/
├── DIV2K/
└── Flickr2K/
```

不要混入验证集、测试集或已加噪图像。默认先裁剪 256×256，再生成同尺寸训练样本；小于初始裁剪尺寸的图像会先放大。

### SIDD 训练与验证

```text
datasets/SIDD_medium/noisy/
datasets/SIDD_medium/gt/
testsets/sidd_validation/noisy/
testsets/sidd_validation/gt/
```

使用对齐的 **8-bit sRGB** 图像。两侧相对文件名应一致，或使用官方 SIDD 的 NOISY/GT 命名。支持的示例：

```text
noisy/0001_NOISY_SRGB_010.PNG
gt/0001_GT_SRGB_010.PNG

noisy/ValidationNoisyBlocksSrgb_0001_01.png
gt/ValidationGtBlocksSrgb_0001_01.png
```


## 4. 配置文件：先修改哪些参数

两个模板分别是：

- [高斯噪声配置](../options/train_ant_blind_denoising.json)
- [SIDD 配置](../options/train_ant_sidd_HL.json)

| 参数 | 修改建议 |
|:--|:--|
| `task` | 每个独立实验使用新名称，避免自动续训旧实验 |
| `gpu_ids` | 默认 [0]；双卡示例 [0,1]，表示物理 GPU 编号 |
| `path.root` | 默认 experiments，确保输出盘有足够空间 |
| `datasets.train.dataroot_H/L` | 修改为本机训练数据路径 |
| `datasets.test.dataroot_H/L` | 修改为独立验证集路径 |
| `pre_trained.pre_trained_diff_path` | 指向扩散主干权重 |
| `dataloader_batch_size` | 默认单卡总 batch=2；论文是双卡总 batch=4 |
| `dataloader_num_workers` | Windows 从 0 开始；Linux 可尝试总 workers=4 或 8 |
| `train.total_iter` | 发布模板默认 600000，可按实验预算调整 |
| `checkpoint_test/save/print` | 均按 iteration 计数，必须大于 0 |

这是带 `//` 注释的 JSON，必须用项目解析器读取，不能直接当作严格 JSON 解析。路径相对于运行命令时的工作目录。Windows 建议写成 `D:/datasets/...`，不要写未转义的反斜杠；简单注释解析器不支持字符串内的 `//`。

`netG.dim`、`nc_config`、`t_config`、`input_resolution` 必须与权重一致。`input_resolution=128` 是构建网络时的注意力参数，不代表训练裁剪尺寸；训练裁剪由 `lq_patchsize=256` 控制。修改裁剪尺寸、batch 或噪声范围可能改变结果。

## 5. 开始训练

```bash
python main_train_antd.py --opt options/train_ant_blind_denoising.json
```

或：

```bash
python main_train_antd.py --opt options/train_ant_sidd_HL.json
```

冻结扩散模型意味着不更新其参数，**不意味着训练时可以对它使用 `torch.no_grad()`**。重建损失仍需通过扩散主干回传至 ANT。

### 保存、恢复与微调

输出位于 `experiments/<task>/`。模型为 `models/<iteration>_G.pth`，优化器为同 iteration 的 `*_optimizerG.pth`。还会保存训练日志、解析后的配置和验证图像。

- **从头训练**：使用新的 task，`pretrained_netG=null`。
- **继续训练**：保留 task，`auto_resume=true`、`G_optimizer_reuse=true`，保存的 G 与 optimizer 必须是同一步。
- **微调**：使用新的 task，`auto_resume=false`，指定 `pretrained_netG`，`pretrained_optimizerG=null`。

到达 `total_iter` 后自动停止并保存末次模型。续训恢复权重、Adam 状态和相应迭代的学习率；不恢复 RNG 状态与数据迭代位置，因此不能声称逐位一致恢复。

## 6. 测试与指标

### 高斯盲去噪

```bash
python main_test_antd_blind_denoising.py --testset_name cbsd68 --noise_level_img 50 --noise_seed 0 --device cuda
```

可换用 `kodak24`、`McMaster`、`Urban100`，并依次测试 15、25、50、100、150、200。注意目录大小写。输入必须是干净图像，脚本负责加噪；σ只用于构造测试数据，不作为 ANT 输入。输出位于 `results/antd_denoising/<dataset>_<model>_<sigma>/`。

### SIDD RGB

```bash
python main_test_antd_sidd_rgb.py --model_name ANT_sidd --L_path testsets/sidd_validation/noisy --H_path testsets/sidd_validation/gt --device cuda
```

输出位于 `results/sidd_ANT_sidd/`。该入口要求 GT，用于完整指标评价。自定义权重通过 `--model_zoo` 指定文件夹，通过 `--model_name` 指定不含 `.pth` 的文件名。

### 无参考评价

```bash
python test_avg_iqa.py --image_folder results/sidd_ANT_sidd --metrics maniqa clipiqa musiq --device cuda
```

输入文件夹只能包含复原图像，不要混入 noisy/GT。程序递归搜索图片并报告每项指标成功评价的张数；如果有失败，只对成功子集求均值，该结果不能直接作为全数据集得分。
