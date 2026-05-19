#!/usr/bin/env bash
# =============================================================================
# Ablation study for Table 4 of the paper (CIFAR-10 IID).
#
# Each ablation removes/replaces one component of SIFS.  Variants override
# specific SIFS flags so the rest of the recipe is held fixed.
# =============================================================================

set -e
DATA_ROOT=${DATA_ROOT:-/home/user/dataset}
GPU=${GPU:-0}

COMMON="--n_agents 10 --n_joined 5 --dir_data $DATA_ROOT \
        --data_train cifar10 --data_test cifar10 --split iid \
        --model_type snn --T 10 --local_epochs 2 --batch_size 32 \
        --epochs 500 --decay step-250-375 --lr 0.1 \
        --fraction_list 0.25,0.5,0.75,1 --template ResNet18 \
        --model spiking_resnet_flanc --basis_fraction 0.125 --n_basis 0.25 \
        --sifs --sifs_init_sparsity 0.0 --sifs_target_sparsity 0.9 \
        --sifs_warmup_rounds 5 --sifs_mask_update_interval 2 \
        --sifs_rebirth_ratio 0.3 --sifs_silence_threshold 1e-3 \
        --sifs_silence_weight 10.0 --sifs_taylor_weight 1.0 \
        --sifs_crisis_weight 1e-3 --sifs_crisis_floor 0.02 \
        --sifs_aggregator mask_aware --sifs_log_silence"

# (A) Full SIFS
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --save abl_full

# (B) Magnitude pruning instead of I_Prune
#     Approximated by silence_weight=0 + taylor_weight=1 but with |w_ij|
#     replacing |w*g|. (Requires --sifs_prune_score magnitude; see option.py
#     TODO if not yet implemented; otherwise use taylor-only as proxy below.)
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --sifs_silence_weight 0.0 --sifs_taylor_weight 1.0 \
  --save abl_magnitude_prune

# (C) Taylor-only (remove silence term)
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --sifs_silence_weight 0.0 --sifs_taylor_weight 1.0 \
  --save abl_taylor_only

# (D) No rebirth growth (prune-only)
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --sifs_rebirth_ratio 0.0 --save abl_no_grow

# (E) Naive FedAvg aggregation (no mask-aware support normalisation)
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --sifs_aggregator fedavg --save abl_naive_agg

# (F) No crisis regulariser
CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
  --sifs_crisis_weight 0.0 --save abl_no_crisis

# (G) Sparsity sweep (Figure 2)
for SF in 0.50 0.70 0.80 0.90 0.95 0.97 0.99; do
  CUDA_VISIBLE_DEVICES=$GPU python ../../main_FL.py $COMMON \
    --sifs_target_sparsity $SF --save abl_sf${SF}
done
