#!/usr/bin/env bash
# =============================================================================
# Dirichlet non-IID sweep for Table 2 (CIFAR-10).
# =============================================================================
set -e
DATA_ROOT=${DATA_ROOT:-/home/user/dataset}
GPU=${GPU:-0}

COMMON="--n_agents 10 --n_joined 5 --dir_data $DATA_ROOT \
        --data_train cifar10 --data_test cifar10 --split noiid \
        --model_type snn --T 10 --local_epochs 2 --batch_size 32 \
        --epochs 500 --decay step-250-375 --lr 0.1 \
        --fraction_list 0.25,0.5,0.75,1 --template ResNet18 \
        --model spiking_resnet_flanc --basis_fraction 0.125 --n_basis 0.25"

SIFS_ARGS="--sifs \
           --sifs_init_sparsity 0.0 --sifs_target_sparsity 0.9 \
           --sifs_warmup_rounds 5 --sifs_mask_update_interval 2 \
           --sifs_rebirth_ratio 0.3 --sifs_silence_threshold 1e-3 \
           --sifs_silence_weight 10.0 --sifs_taylor_weight 1.0 \
           --sifs_crisis_weight 1e-3 --sifs_crisis_floor 0.02 \
           --sifs_aggregator mask_aware --sifs_log_silence"

FEDDST_ARGS="--sifs \
             --sifs_init_sparsity 0.0 --sifs_target_sparsity 0.9 \
             --sifs_warmup_rounds 5 --sifs_mask_update_interval 2 \
             --sifs_rebirth_ratio 0.3 \
             --sifs_silence_weight 0.0 --sifs_taylor_weight 1.0 \
             --sifs_crisis_weight 0.0 \
             --sifs_aggregator fedavg"

for ALPHA in 1.0 0.3 0.1; do
  # SIFS (full)
  CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON $SIFS_ARGS \
    --alpha $ALPHA --save sifs_cifar10_a${ALPHA}

  # FedDST -> SNN naive port at the same sparsity target
  CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON $FEDDST_ARGS \
    --alpha $ALPHA --save feddst_snn_cifar10_a${ALPHA}

  # SFedHIFI baseline
  CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
    --tucker --tucker_epochs 225 --alpha $ALPHA --save sfedhifi_cifar10_a${ALPHA}

  # FedAvg baseline
  CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
    --classic fedavg --alpha $ALPHA --save fedavg_cifar10_a${ALPHA}
done
