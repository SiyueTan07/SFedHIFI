#!/usr/bin/env bash
# =============================================================================
# Main experiments for "Spike-Induced Federated Sparsity" (ICLR 2026).
#
# Reproduces Table 1 (IID main results on Fashion-MNIST / CIFAR-10 / CIFAR-100)
# and Table 3 (CIFAR10-DVS / N-Caltech101).
#
# Flags map 1:1 to option.py.  Set DATA_ROOT and GPU to your environment.
# Logs go to ../experiment/<save_name>/.
# =============================================================================

set -e

DATA_ROOT=${DATA_ROOT:-/home/user/dataset}
GPU=${GPU:-0}

# ---------- common FL / FLANC args ----------
COMMON_FL_ARGS="--n_agents 10 --n_joined 5 --model_type snn \
                --fraction_list 0.25,0.5,0.75,1 \
                --template ResNet18 \
                --basis_fraction 0.125 --n_basis 0.25"

# ---------- SIFS args (wired into option.py) ----------
SIFS_ARGS="--sifs \
           --sifs_init_sparsity 0.0 --sifs_target_sparsity 0.9 \
           --sifs_warmup_rounds 5 \
           --sifs_mask_update_interval 2 \
           --sifs_rebirth_ratio 0.3 \
           --sifs_silence_threshold 1e-3 \
           --sifs_silence_weight 10.0 --sifs_taylor_weight 1.0 \
           --sifs_crisis_weight 1e-3 --sifs_crisis_floor 0.02 \
           --sifs_aggregator mask_aware \
           --sifs_log_silence"

echo "==============================================="
echo "Exp 1.1  Fashion-MNIST IID, SIFS"
echo "==============================================="
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py \
  --dir_data $DATA_ROOT \
  --data_train fashion-mnist --data_test fashion-mnist \
  $COMMON_FL_ARGS $SIFS_ARGS \
  --T 10 --local_epochs 2 --batch_size 32 --epochs 500 \
  --decay step-250-375 --lr 0.01 \
  --model spiking_cnn_flanc \
  --save sifs_fmnist_iid

echo "==============================================="
echo "Exp 1.2  CIFAR-10 IID, SIFS"
echo "==============================================="
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py \
  --dir_data $DATA_ROOT \
  --data_train cifar10 --data_test cifar10 \
  $COMMON_FL_ARGS $SIFS_ARGS \
  --split iid --T 10 --local_epochs 2 --batch_size 32 --epochs 500 \
  --decay step-250-375 --lr 0.1 \
  --model spiking_resnet_flanc \
  --save sifs_cifar10_iid

echo "==============================================="
echo "Exp 1.3  CIFAR-100 IID, SIFS"
echo "==============================================="
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py \
  --dir_data $DATA_ROOT \
  --data_train cifar100 --data_test cifar100 \
  $COMMON_FL_ARGS $SIFS_ARGS \
  --split iid --T 10 --local_epochs 2 --batch_size 32 --epochs 800 \
  --decay step-300-575 --lr 0.1 \
  --model spiking_resnet_flanc \
  --save sifs_cifar100_iid

echo "==============================================="
echo "Exp 1.4  CIFAR10-DVS IID, SIFS (T=16)"
echo "==============================================="
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py \
  --dir_data $DATA_ROOT \
  --data_train cifar10-dvs --data_test cifar10-dvs \
  $COMMON_FL_ARGS $SIFS_ARGS \
  --split iid --T 16 --local_epochs 2 --batch_size 16 --epochs 200 \
  --decay step-100-160 --lr 0.05 \
  --model spiking_resnet_flanc \
  --save sifs_dvs_iid

echo "==============================================="
echo "Exp 1.5  N-Caltech101 IID, SIFS (T=16)"
echo "==============================================="
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py \
  --dir_data $DATA_ROOT \
  --data_train n-caltech101 --data_test n-caltech101 \
  $COMMON_FL_ARGS $SIFS_ARGS \
  --split iid --T 16 --local_epochs 2 --batch_size 16 --epochs 200 \
  --decay step-100-160 --lr 0.05 \
  --model spiking_resnet_flanc \
  --save sifs_ncaltech_iid
