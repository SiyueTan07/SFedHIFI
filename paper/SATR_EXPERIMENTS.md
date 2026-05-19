# SATR 实验环境与运行指令

本文档记录 SATR（Spike-Aware Tucker Rank Calibration）相关的环境配置、日志输出位置、debug 命令、主实验命令和 baseline/ablation 命令。

> 主方法：`--satr --satr_baseline full`  
> 原始 SFedHIFI baseline：不加 `--satr` / `--sifs`  
> Element-wise SIFS baseline：`--sifs`

---

## 1. 环境配置教程

### 1.1 推荐硬件

- GPU：RTX 4090 24GB 或同级别 CUDA GPU
- CPU：建议 16 cores 以上
- 内存：建议 64GB 以上
- 磁盘：至少 200GB 可用空间（数据集 + 多组实验日志 + checkpoint）

### 1.2 创建 conda 环境

仓库已有 [`environment.yml`](../environment.yml)，推荐用 conda 创建：

```bash
cd /Users/tansiyue1/Documents/GitHub/SFedHIFI
conda env create -f environment.yml
conda activate sfedhifi
```

如果已有环境，可以更新：

```bash
conda activate sfedhifi
conda env update -f environment.yml --prune
```

### 1.3 pip 安装方式

如果不用 conda，也可以：

```bash
cd /Users/tansiyue1/Documents/GitHub/SFedHIFI
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> 注意：`requirements.txt` 中的 `torch>=2.2` 不一定会自动安装与你机器 CUDA 完全匹配的 wheel。4090 推荐 PyTorch 2.2+ + CUDA 11.8/12.1。

### 1.4 PyTorch CUDA 安装检查

进入环境后运行：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

期望看到：

```text
cuda available: True
gpu: NVIDIA GeForce RTX 4090
```

### 1.5 关键依赖检查

```bash
python - <<'PY'
import torch, torchvision, numpy, matplotlib, tensorly, swanlab
import spikingjelly
print('all core deps imported')
PY
```

### 1.6 数据路径

下面所有命令默认：

```bash
--dir_data /home/user/dataset
```

如果数据在别处，例如：

```bash
/Users/tansiyue1/datasets
```

请把所有命令里的 `--dir_data /home/user/dataset` 替换成你的实际路径。

### 1.7 GPU 指定

单卡 4090：

```bash
export CUDA_VISIBLE_DEVICES=0
```

### 1.8 先跑 smoke test

```bash
python scripts/smoke_test_sifs.py
```

这个会检查：

- SIFS sparsity schedule
- mask-aware aggregation
- element-wise SIFS mask update
- SATR component score
- SATR Top-K component mask
- crisis loss

### 1.9 Debug 短跑

正式跑 500 rounds 前，先跑 3 epochs 确认代码不炸：

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 2 \
  --split iid \
  --T 4 \
  --local_epochs 1 \
  --batch_size 16 \
  --epochs 3 \
  --decay step-2 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline full \
  --satr_retain_ratio 0.5 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save debug_satr_full_cifar10
```

---

## 2. 日志与输出位置

假设 `--save satr_full_cifar10_iid_q05`，输出目录通常是：

```text
experiment/satr_full_cifar10_iid_q05/
```

核心文件：

```text
experiment/<save>/log.txt
experiment/<save>/config.txt
experiment/<save>/model/model_m*_latest.pt
experiment/<save>/results/metrics/metrics.csv
experiment/<save>/results/metrics/metrics.pt
experiment/<save>/results/metrics/test_top1_acc.png
experiment/<save>/results/metrics/test_top1_error.png
experiment/<save>/results/metrics/test_loss.png
experiment/<save>/results/metrics/train_top1_acc.png
experiment/<save>/results/metrics/train_loss.png
```

SATR 诊断文件（需要 `--satr_log_scores`）：

```text
experiment/<save>/results/satr/satr_round_XXXX.pt
```

默认不保存 SATR 图片。只有显式设置 `--satr_plot_interval > 0` 才保存：

```text
experiment/<save>/results/satr/component_mask_heatmap_round_XXXX.png
experiment/<save>/results/satr/score_summary_round_XXXX.png
```

---

## 3. 4090 时间粗估

单张 RTX 4090：

| 实验 | 单 run 粗估 |
|---|---:|
| CIFAR-10, 500 rounds, T=10, SFedHIFI | 8–14 h |
| CIFAR-10, 500 rounds, T=10, SATR | 9–16 h |
| CIFAR-10, 500 rounds, T=10, SIFS | 10–18 h |
| CIFAR-100, 800 rounds, T=10, SATR | 16–28 h |
| CIFAR10-DVS, 200 rounds, T=16, SATR | 9–18 h |

CIFAR-10 单 seed 8 组 ablation 约：

```text
72–128 GPU hours ≈ 3–5.5 天
```

三 seeds + CIFAR100 + DVS + non-IID 会进入 2–4 周量级。

---

## 4. 主方法 SATR 运行指令

### 4.1 CIFAR-10 IID：SATR full

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline full \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_silence_threshold 1e-3 \
  --satr_spike_weight 1.0 \
  --satr_taylor_weight 1.0 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_full_cifar10_iid_q05
```

### 4.2 CIFAR-100 IID：SATR full

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar100 \
  --data_test cifar100 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 360 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 800 \
  --decay step-300-575 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline full \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_silence_threshold 1e-3 \
  --satr_spike_weight 1.0 \
  --satr_taylor_weight 1.0 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_full_cifar100_iid_q05
```

### 4.3 CIFAR10-DVS IID：SATR full

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10-dvs \
  --data_test cifar10-dvs \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 90 \
  --split iid \
  --T 16 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 200 \
  --decay step-100-160 \
  --lr 0.05 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline full \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_silence_threshold 1e-3 \
  --satr_spike_weight 1.0 \
  --satr_taylor_weight 1.0 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_full_dvs_iid_q05
```

---

## 5. CIFAR-10 IID baseline / ablation 指令

### 5.1 原始 SFedHIFI

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --save sfedhifi_cifar10_iid
```

### 5.2 SATR static baseline

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline static \
  --save satr_static_cifar10_iid
```

### 5.3 SATR random

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline random \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_random_cifar10_iid_q05
```

### 5.4 SATR magnitude

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline magnitude \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_magnitude_cifar10_iid_q05
```

### 5.5 SATR taylor

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline taylor \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_taylor_cifar10_iid_q05
```

### 5.6 SATR spike

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline spike \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_silence_threshold 1e-3 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_spike_cifar10_iid_q05
```

### 5.7 SATR no-normalisation

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline no_norm \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_silence_threshold 1e-3 \
  --satr_spike_weight 1.0 \
  --satr_taylor_weight 1.0 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_no_norm_cifar10_iid_q05
```

### 5.8 Element-wise SIFS

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --sifs \
  --sifs_init_sparsity 0.0 \
  --sifs_target_sparsity 0.5 \
  --sifs_warmup_rounds 5 \
  --sifs_mask_update_interval 2 \
  --sifs_rebirth_ratio 0.3 \
  --sifs_silence_threshold 1e-3 \
  --sifs_silence_weight 10.0 \
  --sifs_taylor_weight 1.0 \
  --sifs_aggregator mask_aware \
  --sifs_log_silence \
  --save sifs_element_cifar10_iid_s05
```

---

## 6. Non-IID 指令

### 6.1 CIFAR-10 non-IID alpha=1.0

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split noiid \
  --alpha 1.0 \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline full \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_full_cifar10_noniid_a1_q05
```

### 6.2 CIFAR-10 non-IID alpha=0.3

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split noiid \
  --alpha 0.3 \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline full \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_full_cifar10_noniid_a03_q05
```

### 6.3 CIFAR-10 non-IID alpha=0.1

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split noiid \
  --alpha 0.1 \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline full \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_full_cifar10_noniid_a01_q05
```

---

## 7. Budget sweep

### 7.1 q=0.25

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline full \
  --satr_retain_ratio 0.25 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_full_cifar10_iid_q025
```

### 7.2 q=0.5

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline full \
  --satr_retain_ratio 0.5 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_full_cifar10_iid_q05
```

### 7.3 q=0.75

```bash
python main_FL.py \
  --n_agents 10 \
  --dir_data /home/user/dataset \
  --data_train cifar10 \
  --data_test cifar10 \
  --n_joined 5 \
  --model_type snn \
  --tucker \
  --tucker_epochs 225 \
  --split iid \
  --T 10 \
  --local_epochs 2 \
  --batch_size 16 \
  --epochs 500 \
  --decay step-250-375 \
  --lr 0.1 \
  --fraction_list 0.25,0.5,0.75,1 \
  --template ResNet18 \
  --model spiking_resnet_flanc \
  --basis_fraction 0.125 \
  --n_basis 0.25 \
  --satr \
  --satr_baseline full \
  --satr_retain_ratio 0.75 \
  --satr_init_retain_ratio 1.0 \
  --satr_warmup_rounds 5 \
  --satr_update_interval 1 \
  --satr_log_scores \
  --satr_plot_interval 0 \
  --save satr_full_cifar10_iid_q075
```
