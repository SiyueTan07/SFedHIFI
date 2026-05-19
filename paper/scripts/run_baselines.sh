#!/usr/bin/env bash
# =============================================================================
# Baseline experiments for SIFS (ICLR 2026 submission)
#
# Reproduces baseline rows of Table 1: FedAvg, FedProx, FedNova, HeteroFL,
# FjORD, SFedHIFI (Tucker), dense ANN-FL, plus FedDST -> SNN (DST with
# magnitude prune + FedAvg aggregation, our strongest naive port).
# =============================================================================

set -e
DATA_ROOT=${DATA_ROOT:-/home/user/dataset}
GPU=${GPU:-0}
DATASET=${DATASET:-cifar10}      # cifar10 / cifar100 / fashion-mnist
SPLIT=${SPLIT:-iid}              # iid / noiid

COMMON="--n_agents 10 --n_joined 5 --dir_data $DATA_ROOT \
        --data_train $DATASET --data_test $DATASET \
        --split $SPLIT --T 10 --local_epochs 2 --batch_size 32 \
        --fraction_list 0.25,0.5,0.75,1 --template ResNet18 \
        --model spiking_resnet_flanc --basis_fraction 0.125 --n_basis 0.25"

EPOCHS_CIFAR10=500
EPOCHS_CIFAR100=800
DECAY_CIFAR10="step-250-375"
DECAY_CIFAR100="step-300-575"

if [ "$DATASET" = "cifar100" ]; then EPOCHS=$EPOCHS_CIFAR100; DECAY=$DECAY_CIFAR100; else EPOCHS=$EPOCHS_CIFAR10; DECAY=$DECAY_CIFAR10; fi

# ---------- (1) FedAvg ----------
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --classic fedavg --model_type snn --epochs $EPOCHS --decay $DECAY --lr 0.1 \
  --save fedavg_${DATASET}_${SPLIT}

# ---------- (2) FedProx ----------
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --classic fedprox --model_type snn --epochs $EPOCHS --decay $DECAY --lr 0.1 \
  --save fedprox_${DATASET}_${SPLIT}

# ---------- (3) FedNova ----------
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --classic fednova --model_type snn --epochs $EPOCHS --decay $DECAY --lr 0.1 \
  --save fednova_${DATASET}_${SPLIT}

# ---------- (4) HeteroFL (SNN port) ----------
# Requires --heterofl flag to be added to option.py; falls back to fedavg with
# fraction-conditioned sub-sampling.
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --classic fedavg --heterofl --model_type snn --epochs $EPOCHS --decay $DECAY --lr 0.1 \
  --save heterofl_${DATASET}_${SPLIT}

# ---------- (5) FjORD (SNN port) ----------
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --classic fedavg --fjord --model_type snn --epochs $EPOCHS --decay $DECAY --lr 0.1 \
  --save fjord_${DATASET}_${SPLIT}

# ---------- (6) SFedHIFI (Tucker baseline, our direct predecessor) ----------
TUCKER_EPOCHS_CIFAR10=225
TUCKER_EPOCHS_CIFAR100=360
if [ "$DATASET" = "cifar100" ]; then TUCKER_EP=$TUCKER_EPOCHS_CIFAR100; else TUCKER_EP=$TUCKER_EPOCHS_CIFAR10; fi
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --model_type snn --tucker --tucker_epochs $TUCKER_EP \
  --epochs $EPOCHS --decay $DECAY --lr 0.1 \
  --save sfedhifi_${DATASET}_${SPLIT}

# ---------- (7) Dense ANN-FL (energy upper bound) ----------
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --classic fedavg --model_type ann --epochs $EPOCHS --decay $DECAY --lr 0.1 \
  --save annfl_${DATASET}_${SPLIT}

# ---------- (8) FedDST -> SNN (strongest naive port for our ablation) ----------
# Same target sparsity as SIFS, but importance is pure |w*g| (silence_weight=0)
# and aggregation is naive FedAvg (sifs_aggregator=fedavg).  This row demonstrates
# that the spike-emptiness signal and mask-aware aggregator carry the gain, not
# the federated DST scaffolding itself.
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --model_type snn --epochs $EPOCHS --decay $DECAY --lr 0.1 \
  --sifs \
  --sifs_init_sparsity 0.0 --sifs_target_sparsity 0.9 \
  --sifs_warmup_rounds 5 --sifs_mask_update_interval 2 \
  --sifs_rebirth_ratio 0.3 \
  --sifs_silence_weight 0.0 --sifs_taylor_weight 1.0 \
  --sifs_crisis_weight 0.0 \
  --sifs_aggregator fedavg \
  --save feddst_snn_${DATASET}_${SPLIT}
