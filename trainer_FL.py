"""
@author: RTao
@contact: rantaostd@gmail.com
@file: trainer_FL.py
@time: 2025/02/10 10:07
"""
from collections import defaultdict
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
        else:
            # SFedHIFI
            self.sync(budget_record, agent_joined, agent_budget)

    def test(self):
        epoch = self.epoch
        self.ckp.write_log('\nEvaluation:')
        timer_test = utility.timer()
        self.tester.test_all(self.loader_test, timer_test, self.run, epoch)

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






