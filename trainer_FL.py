"""
@author: RTao
@contact: rantaostd@gmail.com
@file: trainer_FL.py
@time: 2025/02/10 10:07
"""
from collections import defaultdict
import csv
from dataclasses import replace
from email.policy import strict
import math
import random

import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import multi_mode_dot


import utility
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as tu
from tqdm import tqdm
import os
import swanlab
import collections
import copy
from syops import get_model_complexity_info
from torchinfo import summary

from model.sifs_utils import (
    polynomial_sparsity,
    mask_aware_aggregate,
    update_mask,
    apply_sifs_masks,
    iter_sifs_parameters,
    register_sifs_masks,
    compute_satr_component_score,
    component_mask_from_scores,
    component_live_ratio,
)

tl.set_backend('pytorch')
class Trainer():
    def __init__(self, args, loader, agent_list, ckp):
        self.args = args
        self.ckp = ckp #save need modified
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loaders_train = loader.loaders_train
        self.agent_list = agent_list[:-1] #last one is the tester
        self.tester = agent_list[-1]
        self.epoch = 0
        self.total_epochs = args.epochs
        self.merger_epoch = args.merger_epoch
        self.last_epoch = args.tucker_epochs
        self.top_channels_indices = []
        self.anchors = {}
        if 'resnet' in args.model.lower():
            self.conv_id = 'layer4.1.conv2.conv.1.weight'
        else:
            self.conv_id = 'conv2.conv.1.weight'
        self.chose_name = 'Tucker'
        self.common_layers = None
        self.metrics_history = []

        for agent in self.agent_list:
            agent.make_optimizer_all(ckp=ckp)
            agent.make_scheduler_all()
        self.tester.make_optimizer_all(ckp=ckp)
        self.tester.make_scheduler_all()

        if self.args.debug:
            self.run = swanlab.init(mode="local", project=args.project)
        else:
            self.run = swanlab.init(project=args.project)
        self.run.name = args.save

        self.device = torch.device('cpu' if args.cpu else 'cuda')

        # ---- SIFS/SATR bookkeeping -----------------------------------------
        self.sifs_enabled = bool(getattr(args, 'sifs', False))
        self.satr_enabled = bool(getattr(args, 'satr', False))
        self.filter_mask_enabled = self.sifs_enabled or self.satr_enabled
        if self.filter_mask_enabled:
            # Tester must also carry mask buffers so it can hold the global mask.
            for m in self.tester.model_list:
                register_sifs_masks(m)
            tag = "SATR" if self.satr_enabled and not self.sifs_enabled else "SIFS/SATR"
            print(f"[{tag}] Trainer-level filter-bank mask buffers registered on tester (n_tiers={len(self.tester.model_list)}).")
            self._sifs_round = 0
            self._satr_round = 0
            self._satr_server_fb_full = {}

        self.sync_at_init()

        if args.model.find('INQ') >= 0:
            self.inq_steps = args.inq_steps
        else:
            self.inq_steps = None

    def train(self):
        self.epoch += 1
        epoch = self.epoch

        for agent in self.agent_list:
            agent.make_optimizer_all()
            agent.make_scheduler_all(reschedule=epoch-1) #pls make sure if need resume
        # Step 1: sample a list of agent w/o replacement
        agent_joined = np.sort(np.random.choice(range(self.args.n_agents), self.args.n_joined, replace=False))
        # Step 2: sample a list of associated budgets
        while True:
            agent_budget = np.random.choice(self.args.fraction_list, self.args.n_joined)
            # For implementation simiplicity, we sample all model size for all methods
            _, unique_counts = np.unique(agent_budget, return_counts=True)
            if len(unique_counts) == len(self.args.fraction_list):
                break
        # Step 3: create a buget -> client dictionary for syncing later
        budget_record = collections.defaultdict(list) #need to make sure is not empty
        for k, v in zip(agent_budget, agent_joined):
            budget_record[k].append(v)

        for i in agent_joined:
            self.agent_list[i].begin_all(epoch, self.ckp) #call move all to train()
            self.agent_list[i].start_loss_log() #need check!

        self.start_epoch() #get current lr
        timer_model = utility.timer()

        for idx, i in enumerate(agent_joined):
            timer_model.tic()

            loss, loss_orth, log_train = self.agent_list[i].train_local(self.loaders_train[i],
                              agent_budget[idx], self.args.local_epochs)

            timer_model.hold()
            tt = timer_model.release()

            self.ckp.write_log(
                    '{}/{} ({:.0f}%)\t'
                    'agent {}\t'
                    'model {}\t'
                    'NLL: {:.3f}\t'
                    'Top1: {:.2f} / Top5: {:.2f}\t'
                    'Total {:<2.4f}/ Orth: {:<2.5f} '
                    'Time: {:.1f}s'.format(
                        idx+1,
                        len(agent_joined),
                        100.0 * (idx+1) / len(agent_joined),i,agent_budget[idx],
                        # *(log_train),
                        *log_train,
                        loss, loss_orth,
                        tt
                        )
                    )

        for i in agent_joined:
            self.agent_list[i].log_all(self.ckp)
            for loss in self.agent_list[i].loss_list:
                loss.end_log(len(self.loader_train.dataset)*self.args.local_epochs) #should be accurate

        if self.args.classic:
            self.sync_classic(budget_record, agent_joined, agent_budget)
        elif self.satr_enabled:
            # SATR: component-wise spike-aware filter-bank selection
            self.satr_sync(budget_record, agent_joined, agent_budget)
        elif self.sifs_enabled:
            # SIFS: element-wise Spike-Induced Federated Sparsity
            self.sifs_sync(budget_record, agent_joined, agent_budget)
        else:
            # SFedHIFI (Tucker)
            self.sync(budget_record, agent_joined, agent_budget)

    def _collect_metrics(self, epoch):
        """Collect per-capacity loss/error/accuracy into flat row dicts."""
        rows = []
        for model_id, net_f in enumerate(self.args.fraction_list):
            loss_obj = self.tester.loss_list[model_id]
            row = {
                'epoch': int(epoch),
                'model_id': int(model_id),
                'capacity': float(net_f),
            }
            if loss_obj.log_train.shape[0] > 0:
                train = loss_obj.log_train[-1]
                row.update({
                    'train_loss': float(train[0].item()),
                    'train_top1_error': float(train[1].item()),
                    'train_top5_error': float(train[2].item()),
                    'train_top1_acc': float(100.0 - train[1].item()),
                    'train_top5_acc': float(100.0 - train[2].item()),
                })
            else:
                row.update({
                    'train_loss': float('nan'),
                    'train_top1_error': float('nan'),
                    'train_top5_error': float('nan'),
                    'train_top1_acc': float('nan'),
                    'train_top5_acc': float('nan'),
                })
            if loss_obj.log_test.shape[0] > 0:
                test = loss_obj.log_test[-1]
                row.update({
                    'test_loss': float(test[0].item()),
                    'test_top1_error': float(test[1].item()),
                    'test_top5_error': float(test[2].item()),
                    'test_top1_acc': float(100.0 - test[1].item()),
                    'test_top5_acc': float(100.0 - test[2].item()),
                })
            else:
                row.update({
                    'test_loss': float('nan'),
                    'test_top1_error': float('nan'),
                    'test_top5_error': float('nan'),
                    'test_top1_acc': float('nan'),
                    'test_top5_acc': float('nan'),
                })
            rows.append(row)
        return rows

    def _save_metrics(self, epoch):
        """Save metrics as CSV/PT and update summary plots under results/metrics."""
        rows = self._collect_metrics(epoch)
        self.metrics_history.extend(rows)
        out_dir = os.path.join(self.ckp.dir, 'results', 'metrics')
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, 'metrics.csv')
        fieldnames = [
            'epoch', 'model_id', 'capacity',
            'train_loss', 'train_top1_error', 'train_top5_error', 'train_top1_acc', 'train_top5_acc',
            'test_loss', 'test_top1_error', 'test_top5_error', 'test_top1_acc', 'test_top5_acc',
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.metrics_history:
                writer.writerow(row)
        torch.save(self.metrics_history, os.path.join(out_dir, 'metrics.pt'))

        try:
            swan_logs = {}
            for row in rows:
                cap = row['capacity']
                prefix = f"metrics/capacity_{cap}"
                for key in [
                    'train_loss', 'train_top1_error', 'train_top1_acc',
                    'test_loss', 'test_top1_error', 'test_top1_acc',
                ]:
                    swan_logs[f"{prefix}/{key}"] = row[key]
            if swan_logs:
                self.run.log(swan_logs, step=epoch)
        except Exception:
            pass

        try:
            self._plot_metrics(out_dir)
        except Exception as e:
            self.ckp.write_log(f'[Metrics] plotting failed at epoch {epoch}: {e}')

    def _plot_metrics(self, out_dir):
        """Plot train/test loss and accuracy curves per capacity tier."""
        if not self.metrics_history:
            return
        capacities = sorted(set(row['capacity'] for row in self.metrics_history))
        specs = [
            ('test_top1_acc', 'Test Top-1 Accuracy (%)', 'test_top1_acc.png'),
            ('test_top1_error', 'Test Top-1 Error (%)', 'test_top1_error.png'),
            ('test_loss', 'Test Loss', 'test_loss.png'),
            ('train_top1_acc', 'Train Top-1 Accuracy (%)', 'train_top1_acc.png'),
            ('train_loss', 'Train Loss', 'train_loss.png'),
        ]
        for key, title, filename in specs:
            plt.figure(figsize=(7, 5))
            for cap in capacities:
                rows = [r for r in self.metrics_history if r['capacity'] == cap]
                rows = sorted(rows, key=lambda r: r['epoch'])
                xs = [r['epoch'] for r in rows]
                ys = [r[key] for r in rows]
                plt.plot(xs, ys, marker='o', linewidth=1.5, markersize=3, label=f'capacity={cap}')
            plt.xlabel('Epoch')
            plt.ylabel(title)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, filename), dpi=200)
            plt.close()

    def test(self):
        epoch = self.epoch
        self.ckp.write_log('\nEvaluation:')
        timer_test = utility.timer()
        self.tester.test_all(self.loader_test, timer_test, self.run, epoch)
        self._save_metrics(epoch)

    def sync_at_init(self):
        if self.args.resume_from:
            for i in range(len(self.args.fraction_list)):
                print("resume from checkpoint")
                self.tester.model_list[i].load_state_dict(torch.load('../experiment/'+self.args.save+'/model/model_m'+str(i)+'_'+self.args.resume_from+'.pt'))
        # Sync all agents' parameters with tester before training
        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            state_dict = self.tester.model_list[model_id].state_dict()
            for agent in self.agent_list:
                agent.model_list[model_id].load_state_dict(copy.deepcopy(state_dict),strict=True)

    def sync(self, budget_record, agent_joined, agent_budget):
        # Step 1: gather all filter banks
        # This step runs across network fractions
        filter_banks = {}
        for k, v in self.tester.model_list[0].state_dict().items():
            if 'filter_bank' in k:
                filter_banks[k] = torch.zeros(v.shape)

        for k in filter_banks:
            for b, i in zip(agent_budget, agent_joined):
                model_id = self.agent_list[i].budget_to_model(b)
                state_dict = self.agent_list[i].model_list[model_id].state_dict()
                filter_banks[k] += state_dict[k] * (1. / self.args.n_joined)
        # Step 2: gather all other parameters
        # This step runs within each network fraction
        anchors = {}
        for net_f in self.args.fraction_list:
            n_models = len(budget_record[net_f])
            agent_list_at_net_f = budget_record[net_f]

            anchor = {}
            model_id = self.tester.budget_to_model(net_f)
            for k, v in self.tester.model_list[model_id].state_dict().items():
                if 'filter_bank' not in k:
                    anchor[k] = torch.zeros(v.shape)
            for k in anchor:
                for i in agent_list_at_net_f:
                    model_id = self.agent_list[i].budget_to_model(net_f)
                    state_dict = self.agent_list[i].model_list[model_id].state_dict()
                    anchor[k] += state_dict[k] * (1. / n_models)
            anchors[net_f] = anchor

        # [SFedHIFI] Aggregate knowledge shared by anchors of different scales.
        if self.args.tucker and self.epoch <= self.last_epoch and self.epoch % self.merger_epoch == 0:
                anchors = self.tucker_sync(anc=anchors)

        # Step 3: distribute anchors and filter banks to all agents
        for agent in self.agent_list:
            for net_f in self.args.fraction_list:
                model_id = agent.budget_to_model(net_f)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)

        # Last step: update tester
        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)
        return filter_banks, anchors

    def sync_classic(self, budget_record, agent_joined, agent_budget):
        class_name = self.args.classic.lower()
        w_classics = {}
        for net_f in self.args.fraction_list:
            n_models = len(budget_record[net_f])
            agent_list_at_net_f = budget_record[net_f]
            w_classic = {}
            model_id = self.tester.budget_to_model(net_f)
            if class_name == 'fednova':
                tau_lists = []
                models = []
                last_model = copy.deepcopy(self.tester.model_list[model_id])
                for i in agent_list_at_net_f:
                    tau_lists.append(len(self.loaders_train[i]) * self.args.local_epochs)
                    model_id = self.agent_list[i].budget_to_model(net_f)
                    state_dict = self.agent_list[i].model_list[model_id]
                    models.append(state_dict)
                w_classics[net_f] = self.tester.fednova_(last_model=last_model, buffer=[models, tau_lists]).state_dict()

            else:
                for k, v in self.tester.model_list[model_id].state_dict().items():
                    w_classic[k] = torch.zeros(v.shape)
                for k in self.tester.model_list[model_id].state_dict():
                    for i in agent_list_at_net_f:
                        model_id = self.agent_list[i].budget_to_model(net_f)
                        state_dict = self.agent_list[i].model_list[model_id].state_dict()
                        w_classic[k] += state_dict[k] * (1. / n_models)
                w_classics[net_f] = w_classic

        # Distribute anchors and filter banks to all agents
        for agent in self.agent_list:
            for net_f in self.args.fraction_list:
                model_id = agent.budget_to_model(net_f)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(w_classics[net_f]), strict=False)

        # Update tester
        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(w_classics[net_f]), strict=False)
        return w_classics

    def tucker_sync(self, anc):
        print("Tucker decomposition:")
        start = time.time()
        anchors = copy.deepcopy(anc)
        weights_dict = {}
        rankl = {}

        # decompose random/fixed layers
        if self.args.all_tucker:
            if self.args.all_tucker == 'random':
                if self.common_layers == None:
                    net_keys = []
                    # new_weights_dict = {}
                    for net_f in self.args.fraction_list:
                        net_keys.append(set(anchors[net_f].keys()))
                    common_layers = set.intersection(*[set(keys) for keys in net_keys])
                    self.common_layers = [p for p in common_layers if p.endswith('.conv.1.weight')]
                layer_ = random.choice(self.common_layers)
            else:
                if self.args.all_tucker == 'fixed':
                    fixed_conv_id = self.conv_id
                else:
                    fixed_conv_id = self.args.all_tucker
                layer_ = fixed_conv_id
        # decompose the layer [self.tester.layer_fr_id] with highest fire rate
        else:
            layer_ = f"{self.tester.layer_fr_id}{'.conv.1.weight'}" if (self.args.model_type == 'snn' and self.tester.layer_fr_id) else self.conv_id

        for net_f in self.args.fraction_list:
            weights_dict[net_f] = anchors[net_f][layer_].clone().detach().cpu().numpy()
            rankl[net_f] = weights_dict[net_f].shape
        tuples = list(rankl.values())
        rank = [min(dim_vals) for dim_vals in zip(*tuples)]
        rank = [int(round(val * self.args.n_basis)) if i < 2 else val for i, val in enumerate(rank)]

        decomposed_weights = self.tucker_decompose_weights(weights_dict, rank)

        print(f"Layer: {layer_}")
        print(f"rank: {rank}")
        for key, value in decomposed_weights.items():
            print(f"Net scale: {key}")
            print(f"Core tensor: {value['core'].shape}")
            print(f"Factor Matrices: {[factor.shape for factor in value['factors']]}")

        # aggregate the Core tensors and assign them to anchors[net_f][conv].
        aggregated_core = self.aggregate_cores(decomposed_weights)
        new_weights_dict = self.reconstruct_with_aggregated_core(decomposed_weights, aggregated_core)
        for net_f, tensor in new_weights_dict.items():
            anchors[net_f][layer_] = tensor

        duration = time.time() - start
        print('\nTotal Run Time of Tucker Decomposition: {0:0.4f} s'.format(duration))
        return anchors

    # ===================================================================
    # SATR: Spike-Aware Tucker Rank/Component Calibration sync
    # ===================================================================
    def _aggregate_anchors(self, budget_record):
        """FedAvg non-filter-bank parameters within each capacity tier."""
        anchors = {}
        for net_f in self.args.fraction_list:
            n_models = len(budget_record[net_f])
            agent_list_at_net_f = budget_record[net_f]
            anchor = {}
            model_id = self.tester.budget_to_model(net_f)
            for k, v in self.tester.model_list[model_id].state_dict().items():
                if 'filter_bank' not in k and 'mask_filter_bank' not in k:
                    anchor[k] = torch.zeros(v.shape)
            if n_models == 0:
                anchors[net_f] = anchor
                continue
            for k in anchor:
                for i in agent_list_at_net_f:
                    model_id = self.agent_list[i].budget_to_model(net_f)
                    sd = self.agent_list[i].model_list[model_id].state_dict()
                    anchor[k] += sd[k] * (1.0 / n_models)
            anchors[net_f] = anchor
        return anchors

    def _average_client_spike_rates(self, agent_joined):
        """Average last-round spike rates reported by selected clients."""
        client_rates = []
        for i in agent_joined:
            cs = self.agent_list[i].get_sifs_silence()
            if cs:
                client_rates.append(cs)
        avg_rates = {}
        if not client_rates:
            return avg_rates
        all_keys = set()
        for cs in client_rates:
            all_keys.update(cs.keys())
        for k in all_keys:
            pieces = [cs[k] for cs in client_rates if k in cs]
            if pieces:
                L = min(p.numel() for p in pieces)
                avg_rates[k] = torch.stack([p[:L] for p in pieces], dim=0).mean(dim=0)
        return avg_rates

    def _match_presyn_rate(self, avg_rates, weight):
        """Heuristic mapping from recorded spike vectors to filter-bank basis_size."""
        for _, rate in avg_rates.items():
            if rate.numel() == weight.shape[1]:
                return rate
        for _, rate in avg_rates.items():
            if rate.numel() == weight.shape[0]:
                return rate
        return None

    def _save_satr_visuals(self, score_log, agg_mask, retain_ratio):
        """Persist SATR diagnostics and optional visualisations to disk."""
        if not getattr(self.args, 'satr_log_scores', False):
            return
        out_dir = os.path.join(self.ckp.dir, 'results', 'satr')
        os.makedirs(out_dir, exist_ok=True)
        round_id = getattr(self, '_satr_round', self.epoch)

        serialisable = {
            'round': round_id,
            'epoch': self.epoch,
            'retain_ratio': retain_ratio,
            'score_log': score_log,
            'component_masks': {
                k: v.detach().cpu().view(v.shape[0], -1).mean(dim=1)
                for k, v in agg_mask.items()
            },
        }
        torch.save(serialisable, os.path.join(out_dir, f'satr_round_{round_id:04d}.pt'))

        plot_interval = int(getattr(self.args, 'satr_plot_interval', 10))
        if plot_interval <= 0 or round_id % plot_interval != 0:
            return

        try:
            keys = list(agg_mask.keys())
            if not keys:
                return
            max_components = max(agg_mask[k].shape[0] for k in keys)
            heat = torch.full((len(keys), max_components), float('nan'))
            for row, k in enumerate(keys):
                comp = agg_mask[k].detach().float().view(agg_mask[k].shape[0], -1).mean(dim=1)
                heat[row, :comp.numel()] = comp.cpu()

            fig_h = max(4, 0.28 * len(keys))
            plt.figure(figsize=(12, fig_h))
            plt.imshow(heat.numpy(), aspect='auto', interpolation='nearest', vmin=0, vmax=1, cmap='viridis')
            plt.colorbar(label='component active (0/1)')
            plt.yticks(range(len(keys)), keys, fontsize=6)
            plt.xlabel('component index')
            plt.title(f'SATR component mask heatmap | round={round_id} | retain={retain_ratio:.3f}')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'component_mask_heatmap_round_{round_id:04d}.png'), dpi=200)
            plt.close()

            means = [score_log[k]['score_mean'] for k in keys if k in score_log]
            stds = [score_log[k]['score_std'] for k in keys if k in score_log]
            labels = [k for k in keys if k in score_log]
            if labels:
                plt.figure(figsize=(12, max(4, 0.28 * len(labels))))
                y = list(range(len(labels)))
                plt.barh(y, means, xerr=stds, color='steelblue', alpha=0.85)
                plt.yticks(y, labels, fontsize=6)
                plt.xlabel('component score mean ± std')
                plt.title(f'SATR score summary | round={round_id}')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'score_summary_round_{round_id:04d}.png'), dpi=200)
                plt.close()
        except Exception as e:
            self.ckp.write_log(f'[SATR] visualisation failed at round {round_id}: {e}')

    def _satr_retain_ratio(self):
        """Polynomial schedule for retained component fraction q(t)."""
        rnd = self._satr_round
        if rnd <= self.args.satr_warmup_rounds:
            return float(self.args.satr_init_retain_ratio)
        if rnd >= self.args.satr_final_round:
            return float(self.args.satr_retain_ratio)
        progress = (rnd - self.args.satr_warmup_rounds) / max(1, self.args.satr_final_round - self.args.satr_warmup_rounds)
        frac = (1.0 - progress) ** 3.0
        return float(self.args.satr_retain_ratio + (self.args.satr_init_retain_ratio - self.args.satr_retain_ratio) * frac)

    def satr_sync(self, budget_record, agent_joined, agent_budget):
        """Component-wise filter-bank aggregation and selection.

        Baselines are selected by ``--satr_baseline``:
          static      -> original SFedHIFI sync
          random      -> random Top-K components
          magnitude   -> Top-K by component norm
          taylor      -> Top-K by update/Taylor proxy
          spike       -> Top-K by spike utilisation
          no_norm     -> unnormalised spike + Taylor
          full        -> normalised spike + Taylor (SATR)
        """
        mode = getattr(self.args, 'satr_baseline', 'full')
        if mode == 'static':
            return self.sync(budget_record, agent_joined, agent_budget)

        self._satr_round += 1
        retain_ratio = self._satr_retain_ratio()
        do_update = (
            self._satr_round > self.args.satr_warmup_rounds
            and (self._satr_round - self.args.satr_warmup_rounds) % self.args.satr_update_interval == 0
        )
        print(f"[SATR] round={self._satr_round} mode={mode} retain={retain_ratio:.3f} update={do_update}")

        anchors = self._aggregate_anchors(budget_record)
        avg_rates = self._average_client_spike_rates(agent_joined)

        ref_state = self.tester.model_list[0].state_dict()
        fb_keys = [k for k in ref_state.keys() if k.endswith('filter_bank_1') or k.endswith('filter_bank_2')]
        full_store = getattr(self, '_satr_server_fb_full', {})
        agg_fb = {}
        agg_mask = {}
        score_log = {}

        for fb_key in fb_keys:
            mask_key = fb_key.replace('filter_bank_', 'mask_filter_bank_')
            ref_w = full_store.get(fb_key, ref_state[fb_key]).detach().clone().cpu()
            ref_mask = ref_state.get(mask_key, torch.ones_like(ref_w)).detach().clone().cpu()

            weighted_sum = torch.zeros_like(ref_w)
            coverage = torch.zeros_like(ref_w)
            score_sum = None
            score_count = 0

            for b, i in zip(agent_budget, agent_joined):
                model_id = self.agent_list[i].budget_to_model(b)
                sd = self.agent_list[i].model_list[model_id].state_dict()
                if fb_key not in sd:
                    continue
                w = sd[fb_key].detach().clone().cpu()
                m = sd.get(mask_key, torch.ones_like(w)).detach().clone().cpu()
                weighted_sum += w * m
                coverage += m

                presyn_rate = self._match_presyn_rate(avg_rates, w)
                score = compute_satr_component_score(
                    w,
                    reference_weight=ref_w,
                    presyn_rate=presyn_rate,
                    mode=mode,
                    silence_threshold=self.args.satr_silence_threshold,
                    spike_weight=self.args.satr_spike_weight,
                    taylor_weight=self.args.satr_taylor_weight,
                    normalise=(mode not in ('no_norm', 'no-normalisation', 'no-normalization')),
                ).cpu()
                score_sum = score if score_sum is None else score_sum + score
                score_count += 1

            # Preserve inactive components from previous server state.
            agg = ref_w.clone()
            live = coverage > 0
            agg[live] = weighted_sum[live] / (coverage[live] + 1e-12)

            if score_sum is None:
                scores = torch.ones(ref_w.shape[0])
            else:
                scores = score_sum / max(1, score_count)

            if do_update:
                new_mask = component_mask_from_scores(ref_w, scores, retain_ratio=retain_ratio).cpu()
            else:
                new_mask = ref_mask

            # Keep a full server-side copy for future Taylor/update scoring,
            # but only broadcast active components to clients/tester models.
            full_store[fb_key] = agg.clone()
            agg_fb[fb_key] = agg * new_mask
            agg_mask[mask_key] = new_mask
            score_log[fb_key] = {
                'live_ratio': component_live_ratio(new_mask),
                'score_mean': float(scores.float().mean().item()),
                'score_std': float(scores.float().std(unbiased=False).item()) if scores.numel() > 1 else 0.0,
            }

        self._satr_server_fb_full = full_store

        global_fb_state = {}
        global_fb_state.update(agg_fb)
        global_fb_state.update(agg_mask)

        # Keep SFedHIFI's optional Tucker/anchor merger unchanged.
        if self.args.tucker and self.epoch <= self.last_epoch and self.epoch % self.merger_epoch == 0:
            anchors = self.tucker_sync(anc=anchors)

        for agent in self.agent_list:
            for net_f in self.args.fraction_list:
                model_id = agent.budget_to_model(net_f)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(global_fb_state), strict=False)
                apply_sifs_masks(agent.model_list[model_id])

        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(global_fb_state), strict=False)
            apply_sifs_masks(self.tester.model_list[model_id])

        if getattr(self.args, 'satr_log_scores', False):
            try:
                logs = {}
                for k, stats in score_log.items():
                    safe_k = k.replace('.', '/')
                    logs[f"satr/{safe_k}/live_ratio"] = stats['live_ratio']
                    logs[f"satr/{safe_k}/score_mean"] = stats['score_mean']
                    logs[f"satr/{safe_k}/score_std"] = stats['score_std']
                logs["satr/retain_ratio"] = retain_ratio
                self.run.log(logs, step=self.epoch - 1)
            except Exception:
                pass
            self._save_satr_visuals(score_log, agg_mask, retain_ratio)

        return agg_fb, anchors, agg_mask

    # SIFS: Spike-Induced Federated Sparsity sync
    # ===================================================================
    def sifs_sync(self, budget_record, agent_joined, agent_budget):
        """Mask-aware federated aggregation + dynamic prune/grow.

        Per round:
          1. Aggregate filter_banks with the SIFS mask-aware rule.
          2. Aggregate the (per-tier) anchor parameters with vanilla FedAvg
             (these are NOT shared across tiers and do not carry masks).
          3. Compute the target sparsity s(t) via polynomial schedule.
          4. If update interval, run prune+grow on the *global* filter_banks
             using the aggregated weight, surrogate gradient (proxy: weight
             magnitude on the global model), and aggregated presyn rates.
          5. Distribute updated weights + masks back to all clients.
        """
        self._sifs_round += 1
        rnd = self._sifs_round
        do_update = (
            rnd > self.args.sifs_warmup_rounds
            and (rnd - self.args.sifs_warmup_rounds) % self.args.sifs_mask_update_interval == 0
        )

        # ---- Step 1: collect (filter_bank, mask) per client ---------------
        # Filter banks are shared across tiers in FLANC, so we gather them
        # over ALL participating clients regardless of tier.
        # Structure: {fb_param_name: list of (weight_tensor, mask_tensor)}
        fb_buckets = collections.defaultdict(list)

        # Aggregate per-tier non-filter-bank anchors (FedAvg as before)
        anchors = {}
        for net_f in self.args.fraction_list:
            n_models = len(budget_record[net_f])
            agent_list_at_net_f = budget_record[net_f]
            anchor = {}
            model_id = self.tester.budget_to_model(net_f)
            for k, v in self.tester.model_list[model_id].state_dict().items():
                if 'filter_bank' not in k and 'mask_filter_bank' not in k:
                    anchor[k] = torch.zeros(v.shape)
            for k in anchor:
                for i in agent_list_at_net_f:
                    sd = self.agent_list[i].model_list[
                        self.agent_list[i].budget_to_model(net_f)
                    ].state_dict()
                    anchor[k] += sd[k] * (1.0 / n_models)
            anchors[net_f] = anchor

        # Gather filter_banks with their per-client masks
        # We iterate over the tester's first tier as the structural reference,
        # but since filter_bank_{1,2} buffers are shape-identical across tiers
        # (they only depend on n_basis/basis_size, not net_fraction), this works.
        ref_state = self.tester.model_list[0].state_dict()
        fb_keys = [k for k in ref_state.keys() if k.endswith('filter_bank_1') or k.endswith('filter_bank_2')]

        for fb_key in fb_keys:
            mask_key = fb_key.replace('filter_bank_', 'mask_filter_bank_')
            for b, i in zip(agent_budget, agent_joined):
                model_id = self.agent_list[i].budget_to_model(b)
                sd = self.agent_list[i].model_list[model_id].state_dict()
                if fb_key not in sd:
                    continue
                w = sd[fb_key].clone().cpu()
                if mask_key in sd:
                    m = sd[mask_key].clone().cpu()
                else:
                    m = torch.ones_like(w)
                fb_buckets[fb_key].append((w, m))

        # ---- Step 2: mask-aware aggregation -------------------------------
        agg_fb = {}
        agg_mask = {}
        for fb_key, items in fb_buckets.items():
            tensors = [it[0] for it in items]
            masks = [it[1] for it in items]
            if self.args.sifs_aggregator == 'mask_aware':
                agg = mask_aware_aggregate(tensors, masks)
            else:
                # vanilla FedAvg fallback
                stacked = torch.stack(tensors, dim=0)
                agg = stacked.mean(dim=0)
            agg_fb[fb_key] = agg
            # Union mask: an entry is alive globally if ANY client kept it alive
            stacked_m = torch.stack(masks, dim=0)
            agg_mask[fb_key.replace('filter_bank_', 'mask_filter_bank_')] = (
                stacked_m.sum(dim=0) > 0
            ).to(agg.dtype)

        # ---- Step 3: sparsity schedule ------------------------------------
        target_s = polynomial_sparsity(
            current_round=rnd,
            warmup=self.args.sifs_warmup_rounds,
            final_round=self.args.sifs_final_round,
            init_sparsity=self.args.sifs_init_sparsity,
            final_sparsity=self.args.sifs_target_sparsity,
        )
        print(f"[SIFS] round={rnd} target_sparsity={target_s:.3f} update={do_update}")

        # ---- Step 4: prune + grow on global filter_banks ------------------
        if do_update:
            # Collect per-LIF mean silence from clients, average across clients
            # for the same layer name, then broadcast to the filter banks
            # they feed into. We use a coarse heuristic: map LIF layer name
            # to the filter_bank in the *same* BasicBlock.
            client_silences = []
            for i in agent_joined:
                cs = self.agent_list[i].get_sifs_silence()
                if cs:
                    client_silences.append(cs)

            avg_silence = {}
            if client_silences:
                all_keys = set()
                for cs in client_silences:
                    all_keys.update(cs.keys())
                for k in all_keys:
                    pieces = [cs[k] for cs in client_silences if k in cs]
                    if pieces:
                        # average over clients (truncate to min length to be safe)
                        L = min(p.numel() for p in pieces)
                        avg_silence[k] = torch.stack([p[:L] for p in pieces], dim=0).mean(dim=0)

            for fb_key, w in agg_fb.items():
                mask_key = fb_key.replace('filter_bank_', 'mask_filter_bank_')
                current_mask = agg_mask[mask_key]
                # surrogate gradient proxy: magnitude (no global backward here)
                grad_proxy = w.detach().abs()
                # try to fetch a presyn rate matching this block's sn1/sn2
                # heuristic: pick first silence rate whose channel count
                # matches the basis_size dim of this filter bank
                presyn_rate = None
                for sname, srate in avg_silence.items():
                    if srate.numel() == w.shape[1]:
                        presyn_rate = srate
                        break

                new_mask = update_mask(
                    current_mask,
                    weight=w,
                    grad=grad_proxy,
                    presyn_rate=presyn_rate,
                    target_sparsity=target_s,
                    rebirth_ratio=self.args.sifs_rebirth_ratio,
                    silence_weight=self.args.sifs_silence_weight,
                    taylor_weight=self.args.sifs_taylor_weight,
                    silence_threshold=self.args.sifs_silence_threshold,
                )
                agg_mask[mask_key] = new_mask
                agg_fb[fb_key] = w * new_mask

        # ---- Step 5: distribute back --------------------------------------
        # combined dict to load_state_dict (strict=False handles missing keys per tier)
        global_fb_state = {}
        global_fb_state.update(agg_fb)
        global_fb_state.update(agg_mask)

        for agent in self.agent_list:
            for net_f in self.args.fraction_list:
                model_id = agent.budget_to_model(net_f)
                agent.model_list[model_id].load_state_dict(
                    copy.deepcopy(anchors[net_f]), strict=False)
                agent.model_list[model_id].load_state_dict(
                    copy.deepcopy(global_fb_state), strict=False)
                if self.sifs_enabled:
                    apply_sifs_masks(agent.model_list[model_id])

        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            self.tester.model_list[model_id].load_state_dict(
                copy.deepcopy(anchors[net_f]), strict=False)
            self.tester.model_list[model_id].load_state_dict(
                copy.deepcopy(global_fb_state), strict=False)
            if self.sifs_enabled:
                apply_sifs_masks(self.tester.model_list[model_id])

        # Log live ratio per filter bank
        if self.args.sifs_log_silence:
            try:
                live_ratios = {
                    k: float((v > 0).float().mean().item())
                    for k, v in agg_mask.items()
                }
                self.run.log({f"sifs/{k}_live": v for k, v in live_ratios.items()},
                             step=self.epoch - 1)
                self.run.log({"sifs/target_sparsity": target_s}, step=self.epoch - 1)
            except Exception:
                pass

        return agg_fb, anchors, agg_mask

    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()
            return x

        return [_prepare(a) for a in args]

    def start_epoch(self):

        lr = self.agent_list[0].scheduler_list[0].get_lr()[0]

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2}'.format(self.epoch, lr))

        return self.epoch, lr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.epoch
            return epoch >= self.args.epochs

    def _analysis(self):
        flops = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules()
        ])
        flops_conv = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules() if isinstance(m, nn.Conv2d)
        ])
        flops_ori = torch.Tensor([
            getattr(m, 'flops_original', 0) for m in self.model.modules()
        ])

        print('')
        print('FLOPs: {:.2f} x 10^8'.format(flops.sum() / 10**8))
        print('Compressed: {:.2f} x 10^8 / Others: {:.2f} x 10^8'.format(
            (flops.sum() - flops_conv.sum()) / 10**8 , flops_conv.sum() / 10**8
        ))
        print('Accel - Total original: {:.2f} x 10^8 ({:.2f}x)'.format(
            flops_ori.sum() / 10**8, flops_ori.sum() / flops.sum()
        ))
        print('Accel - 3x3 original: {:.2f} x 10^8 ({:.2f}x)'.format(
            (flops_ori.sum() - flops_conv.sum()) / 10**8,
            (flops_ori.sum() - flops_conv.sum()) / (flops.sum() - flops_conv.sum())
        ))
        input()

    def sync_firerate(self):

        conv = self.conv_id
        epoch = self.epoch

        weights_dict = {}
        anchors = self.anchors
        for net_f in self.args.fraction_list:
            weights_dict[net_f] = anchors[net_f][conv].clone().detach().cpu().numpy()

        # Rank of Tucker decomposition
        rank = [32, 64, 1, 1]

        decomposed_weights = self.tucker_decompose_weights(weights_dict, rank)

        for key, value in decomposed_weights.items():
            print(f"Net scale: {key}")
            print(f"Core tensor: {value['core'].shape}")
            print(f"Factor Matrices: {[factor.shape for factor in value['factors']]}")

        # aggregate the Core tensors and assign them to anchors[net_f][conv].
        aggregated_core = self.aggregate_cores(decomposed_weights)
        new_weights_dict = self.reconstruct_with_aggregated_core(decomposed_weights, aggregated_core)

        for net_f, tensor in new_weights_dict.items():
            anchors[net_f][conv] = tensor

        for net_f, layers in anchors.items():
            print(f"net_f: {net_f}, conv1 shape: {layers[conv].shape}")

        # Step 3: distribute anchors and filter banks to all agents
        for agent in self.agent_list:
            for net_f in self.args.fraction_list:
                model_id = agent.budget_to_model(net_f)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)

        # Last step: update tester
        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
            print(f"Distributed to {net_f} by {self.chose_name}: done")
        return anchors

    def tucker_decompose_weights(self,weights_dict, rank):
        """
        Perform Tucker decomposition on each tensor in the dictionary.

        Args:
            weights_dict (dict): Dictionary containing tensors, with keys 1, 0.75, 0.5, and 0.25.
            rank (list): Tucker ranks, e.g., [32, 64, 1, 1].

        Returns:
            decomposed_dict (dict): Dictionary containing the decomposition results
            for each tensor, including the core tensor and factor matrices.
        """
        decomposed_dict = {}

        for key, tensor in weights_dict.items():
            # Check if the tensor is a numpy.ndarray; if so, convert it to torch.Tensor while preserving the data type
            if isinstance(tensor, np.ndarray):
                if tensor.dtype == np.float64:
                    tensor = torch.from_numpy(tensor).to(torch.float64)
                elif tensor.dtype == np.float32:
                    tensor = torch.from_numpy(tensor).to(torch.float32)
                elif tensor.dtype == np.float16:
                    tensor = torch.from_numpy(tensor).to(torch.float16)
                elif tensor.dtype == np.int64:
                    tensor = torch.from_numpy(tensor).to(torch.int64)
                elif tensor.dtype == np.int32:
                    tensor = torch.from_numpy(tensor).to(torch.int32)
                else:
                    tensor = torch.from_numpy(tensor)

            core, factors = tucker(tensor, rank=rank)
            decomposed_dict[key] = {
                'core': core,
                'factors': factors
            }

        return decomposed_dict

    def aggregate_cores(self, decomposed_weights):
        """
        Aggregate the core tensors in decomposed_weights to produce a new core tensor.

        Args:
            decomposed_weights (dict): Dictionary of Tucker decomposition results,
                containing cores and factors.

        Returns:
            aggregated_core (Tensor): The aggregated core tensor.
        """
        cores = [value['core'] for value in decomposed_weights.values()]
        aggregated_core = torch.mean(torch.stack(cores), dim=0)

        return aggregated_core

    def reconstruct_with_aggregated_core(self,decomposed_weights, aggregated_core):
        """
        Reconstruct tensors using the aggregated core tensor and the original factor matrices,
        and generate a new weights dictionary.

        Args:
            decomposed_weights (dict): Dictionary of Tucker decomposition results,
                containing cores and factors.
            aggregated_core (Tensor): The aggregated core tensor.

        Returns:
            new_weights_dict (dict): Dictionary containing the reconstructed tensors.
        """
        new_weights_dict = {}

        for key, value in decomposed_weights.items():
            # Reconstruct using the aggregated core tensor and the corresponding factor matrices
            factors = value['factors']
            reconstructed_tensor = multi_mode_dot(aggregated_core, factors, modes=[0, 1, 2, 3])

            new_weights_dict[key] = reconstructed_tensor

        return new_weights_dict






