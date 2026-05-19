"""
@author: RTao
@contact: rantaostd@gmail.com
@file: agent_federation.py
@time: 2025/02/10 10:07
"""
import os
from importlib import import_module
from sched import scheduler

import numpy as np
import torch
import torch.nn as nn
from IPython import embed
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.utils as tu
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
from spikingjelly.activation_based import functional, monitor, neuron
import torch.nn.functional as F
from collections import Counter

from model.sifs_utils import (
    SpikeSilenceTracker,
    register_sifs_masks,
    apply_sifs_masks,
    gather_sifs_state,
    crisis_loss,
)


class Agent:
    def __init__(self, *args):
        super(Agent, self).__init__()
        print('Init Agent {} and making models...'.format(args[2]))

        self.args = args[0] # args should contain slim rate
        self.ckp = args[1]
        self.my_id = args[2]
        self.crop = self.args.crop
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.precision = self.args.precision
        self.n_GPUs = self.args.n_GPUs
        self.save_models = self.args.save_models
        self.fractions = self.args.fraction_list
        self.top_channels_indices = None
        self.layer_fr_id = None # the highest fire rate layer using for HIFI module

        print("Init a List of Models")
        model_list = []
        self.budget_model = {net_f:i for i,net_f in enumerate(self.fractions)}

        for net_f in self.fractions:
            module = import_module('model.' + self.args.model.lower())
            self.module = module
            new_args = self.args
            new_args.net_fraction = net_f
            model_list.append(module.make_model(new_args))
        self.model_list = model_list

        # ---- SIFS/SATR: register mask buffers on filter banks (no-op if disabled)
        self.sifs_enabled = bool(getattr(self.args, 'sifs', False))
        self.satr_enabled = bool(getattr(self.args, 'satr', False))
        self.filter_mask_enabled = self.sifs_enabled or self.satr_enabled
        self.sifs_silence_rates = {}  # last computed {layer_name: tensor[C]}
        if self.filter_mask_enabled:
            n_masks = 0
            for m in self.model_list:
                n_masks += register_sifs_masks(m)
            tag = "SATR" if self.satr_enabled and not self.sifs_enabled else "SIFS/SATR"
            print(f"[{tag}] Registered {n_masks} filter-bank mask buffers across {len(self.model_list)} capacity tiers.")

        print("Filter bank synced at initalization!")
        self.sync_at_init()
        if not self.args.cpu:
            print('CUDA is ready!')
            torch.cuda.manual_seed(self.args.seed)

        # temporarily disable data parallel

        self.load_all(
            self.ckp.dir,
            pretrained=self.args.pretrained,
            load=self.args.load,
            resume=self.args.resume,
            cpu=self.args.cpu
        )

        for i, m in enumerate(self.model_list):
            print(self.get_model(i), file=self.ckp.log_file)

        self.summarize(self.ckp)

    def test_all(self, loader_test, timer_test, run, epoch):
        timer_test = timer_test
        layer_fr = {}

        for i, model in enumerate(self.model_list):
            self.model_list[i] = self.model_list[i].to(self.device)
            self.loss_list[i].start_log(train=False)
            layer_ids = []
            model.eval()
            with torch.no_grad():
                for img, label in tqdm(loader_test, ncols=80):

                    img, label = self.prepare(img, label)
                    torch.cuda.synchronize()
                    timer_test.tic()

                    prediction = model(img)  # (10,500,10)
                    layer_ids.append(test_firerate(model, img))
                    torch.cuda.synchronize()
                    timer_test.hold()

                    self.loss_list[i](prediction, label, train=False)
                    functional.reset_net(model)

            modes = Counter([item[0] for item in layer_ids]).most_common(1)[0][0]
            mode_rates = [item[1] for item in layer_ids if item[0] in modes]
            mean_rate = sum(mode_rates) / len(mode_rates)
            layer_fr[i] = [modes, mean_rate]

            self.loss_list[i].end_log(len(loader_test.dataset), train=False)
            a = self.loss_list[i]
            b = a.log_test
            best = b.min(0)
            self.model_list[i] = self.model_list[i].to('cpu')
            for j, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
                self.ckp.write_log(
                    'model {} {}: {:.3f} (Best: {:.3f} from epoch {})'.format(
                        i,
                        measure,
                        self.loss_list[i].log_test[-1, j],
                        best[0][j],
                        best[1][j] + 1 if len(self.loss_list[i].log_test) == len(self.loss_list[i].log_train) else best[1][j]
                        )
                    )

            run.log({"acc @ {}".format(self.fractions[i]): 100-self.loss_list[i].log_test[-1, self.args.top]},step=epoch-1)
            total_time = timer_test.release()
            is_best = self.loss_list[i].log_test[-1, self.args.top] <= best[0][self.args.top]
            self.ckp.save(self, i, epoch, is_best=is_best)
            self.scheduler_list[i].step()
        self.get_layer_id(layer_fr)

    def get_layer_id(self, layer_fr):
        # Extract all layer names and firing rates
        layers = [v[0] for v in layer_fr.values()]
        rates = {k: v[1] for k, v in layer_fr.items()}

        if len(set(layers)) == 1:
            # Case 1: All layer names are identical
            self.layer_fr_id = layers[0]
        else:
            # Case 2: There is a mode (most frequently occurring layer name)
            layer_counts = Counter(layers)
            max_count = max(layer_counts.values())
            modes = [name for name, count in layer_counts.items() if count == max_count]
            if len(modes) == 1:
                self.layer_fr_id = modes[0]
            else:
            # Case 3: No mode (all layer names are unique) → choose the layer with the highest firing rate
                max_rate_idx = max(rates.items(), key=lambda x: x[1])[0]
                self.layer_fr_id = layer_fr[max_rate_idx][0]

    def budget_to_model(self, budget):
        return self.budget_model[budget]

    def train_local(self, loader_train, budget, epochs):

        model_id = self.budget_to_model(budget)
        loss_list = []
        loss_orth_list = []
        n_samples = 0
        self.model_list[model_id] = self.model_list[model_id].to(self.device)

        # ---- SIFS/SATR bookkeeping ----------------------------------------
        tracker = None
        if self.filter_mask_enabled:
            tracker = SpikeSilenceTracker().attach(self.model_list[model_id])
            tracker.reset()
            # ensure inactive/pruned filter-bank positions are zero before training begins
            apply_sifs_masks(self.model_list[model_id])

        for epoch in range(epochs):
            for batch, (img, label) in enumerate(loader_train):
                img, label = self.prepare(img, label)
                n_samples += img.size(0)

                self.optimizer_list[model_id].zero_grad()

                # Re-apply mask each step so optimizer momentum / weight decay
                # cannot resurrect inactive/pruned positions between updates.
                if self.filter_mask_enabled:
                    apply_sifs_masks(self.model_list[model_id])

                prediction = self.forward(img, model_id)

                loss, _ = self.loss_list[model_id](prediction, label)

                if self.args.no_loss_orth:
                    loss_orth_list.append(0)
                else:
                    loss_orth = self.args.lambdaR*self.module.orth_loss(self.model_list[model_id],self.args,'L2')
                    loss = loss_orth + loss
                    loss_orth_list.append(loss_orth.item())

                # ---- SIFS crisis regularizer (uses running silence stats)
                if self.sifs_enabled and getattr(self.args, 'sifs_crisis_weight', 0.0) > 0.0 and tracker is not None:
                    cl = crisis_loss(
                        tracker,
                        floor=self.args.sifs_crisis_floor,
                        weight=self.args.sifs_crisis_weight,
                    )
                    if cl.requires_grad is False and cl.item() > 0:
                        # crisis_loss is built from a CPU-side running stat
                        # so we add it as a scalar penalty (no autograd path).
                        # It still serves as a diagnostic and as a hard
                        # constraint via early stopping if needed.
                        loss = loss + cl.to(loss.device)

                loss.backward()

                self.optimizer_list[model_id].step()

                # After step: zero out inactive/pruned positions again so masks stay clean
                if self.filter_mask_enabled:
                    apply_sifs_masks(self.model_list[model_id])

                loss_list.append(loss.item())
                functional.reset_net(self.model_list[model_id])

        log_train = self.loss_list[model_id].log_train[-1,:]/n_samples

        # ---- SIFS/SATR: stash per-channel spike stats then detach hooks ----
        if self.filter_mask_enabled and tracker is not None:
            self.sifs_silence_rates = tracker.mean_rates()
            tracker.detach()

        self.model_list[model_id] = self.model_list[model_id].to('cpu')

        return sum(loss_list)/len(loss_list), sum(loss_orth_list)/len(loss_orth_list), log_train

    def get_sifs_state(self, model_id):
        """Return ``{full_param_name: {'weight','mask'}}`` for the given tier."""
        if not self.filter_mask_enabled:
            return {}
        return gather_sifs_state(self.model_list[model_id])

    def get_sifs_silence(self):
        """Return last-round per-LIF-layer mean spike rates (CPU tensors)."""
        return dict(self.sifs_silence_rates)


    def train_one_step(self, img, label):
        loss_list = []
        loss_orth_list = []
        for i, _ in enumerate(self.model_list):
            self.optimizer_list[i].zero_grad()
            prediction = self.forward(img, i)
            loss, _ = self.loss_list[i](prediction, label,)


            loss_orth = self.args.lambdaR*self.module.orth_loss(self.model_list[i],self.args,'L2')
            loss = loss_orth + loss
            loss_orth_list.append(loss_orth.item())

            loss.backward()
            self.optimizer_list[i].step()

            loss_list.append(loss.item())
            functional.reset_net(self.model)

        if self.args.sync: self.sync_filter()

        return loss_list, loss_orth_list

    def sync_at_init(self):
        n_models = len(self.model_list)
        filter_banks = {}
        for k,v in self.model_list[0].state_dict().items():
            if 'filter_bank' in k:
                filter_banks[k] = v

        for i in range(n_models):
            self.model_list[i].load_state_dict(copy.deepcopy(filter_banks), strict=False)


    def sync_filter(self):
        n_models = len(self.model_list)
        filter_banks = {}
        for k,v in self.model_list[0].state_dict().items():
            if 'filter_bank' in k:
                filter_banks[k] = torch.zeros(v.shape).cuda()

        for k in filter_banks:
            for model in self.model_list:
                state_dict = model.state_dict()
                filter_banks[k]+=state_dict[k]*(1./n_models)

        for i in range(n_models):

            self.model_list[i].load_state_dict(copy.deepcopy(filter_banks), strict=False)

    def forward(self, x, i):
        if self.crop > 1:
            b, n_crops, c, h, w = x.size()
            x = x.view(-1, c, h, w)
        x = self.model_list[i](x)

        if self.crop > 1: x = x.view(b, n_crops, -1).mean(1)

        return x

    def get_model(self, i):
        if self.n_GPUs == 1:
            return self.model_list[i]
        else:
            return self.model_list[i].module

    def state_dict_all(self, **kwargs):
        ret = []
        for i, _ in enumerate(self.model_list):
            ret.append(self.state_dict(i))
        return ret

    def state_dict(self, i, **kwargs):
        return self.get_model(i).state_dict(**kwargs)

    def save_all(self, apath, epoch, is_best=False):
        for i, _ in enumerate(self.model_list):
            self.save(i, apath, epoch, is_best)

    def save(self, i, apath, epoch, is_best=False):
        target = self.get_model(i).state_dict()

        conditions = (True, is_best, self.save_models)
        names = ('latest', 'best', '{}'.format(epoch))

        for c, n in zip(conditions, names):
            if c:
                torch.save(
                    target,
                    os.path.join(apath, 'model', 'model_m{}_{}.pt'.format(i,n))
                )

    def load_all(self, apath, pretrained='', load='', resume=-1, cpu=False):
        for i, _ in enumerate(self.model_list):
            self.load(i, apath, pretrained, load, resume, cpu)

    def load(self, i, apath, pretrained='', load='', resume=-1, cpu=False):
        f = None
        if pretrained:
            if pretrained != 'download':
                print('Load pre-trained model from {}'.format(pretrained))
                f = pretrained
        else:
            if load:
                if resume == -1:
                    print('Load model {} after the last epoch'.format(i))
                    resume = 'latest'
                else:
                    print('Load model {} after epoch {}'.format(i,resume))

                f = os.path.join(apath, 'model', 'model_m{}_{}.pt'.format(i,resume))

        if f:
            kwargs = {}
            if cpu:
                kwargs = {'map_location': lambda storage, loc: storage}
            state = torch.load(f, **kwargs)

            self.get_model(i).load_state_dict(state, strict=False)

    def begin_all(self, epoch, ckp):
        for i, _ in enumerate(self.model_list):
            self.begin(i, epoch, ckp)

    def begin(self, i, epoch, ckp):
        self.model_list[i].train()
        m = self.get_model(i)
        if hasattr(m, 'begin'): m.begin(epoch, ckp)

    def start_loss_log(self):
        for loss in self.loss_list:
            loss.start_log() #create a tensor

    def log_all(self, ckp):
        for i, _ in enumerate(self.model_list):
            self.log(i, ckp)

    def log(self, i, ckp):
        m = self.get_model(i)
        if hasattr(m, 'log'): m.log(ckp)

    def summarize(self, ckp):
        for i, _ in enumerate(self.model_list):
            ckp.write_log('# parameters of model {}: {:,}'.format(i,
                sum([p.nelement() for p in self.model_list[i].parameters()])
            ))

            kernels_1x1 = 0
            kernels_3x3 = 0
            kernels_others = 0
            gen = (c for c in self.model_list[i].modules() if isinstance(c, nn.Conv2d))
            for m in gen:
                kh, kw = m.kernel_size
                n_kernels = m.in_channels * m.out_channels
                if kh == 1 and kw == 1:
                    kernels_1x1 += n_kernels
                elif kh == 3 and kw == 3:
                    kernels_3x3 += n_kernels
                else:
                    kernels_others += n_kernels

            linear = sum([
                l.weight.nelement() for l in self.model_list[i].modules() \
                if isinstance(l, nn.Linear)
            ])

            ckp.write_log(
                '1x1: {:,}\n3x3: {:,}\nOthers: {:,}\nLinear:{:,}\n'.format(
                    kernels_1x1, kernels_3x3, kernels_others, linear
                ),
                refresh=True
            )
    def make_optimizer_all(self, ckp=None, lr=None):
        ret = []
        for i, _ in enumerate(self.model_list):
            ret.append(self.make_optimizer(i, ckp, lr))
        self.optimizer_list = ret

    def make_optimizer(self, i, ckp=None, lr=None):
        trainable = filter(lambda x: x.requires_grad, self.model_list[i].parameters())

        if self.args.optimizer == 'SGD':
            optimizer_function = optim.SGD
            kwargs = {'momentum': self.args.momentum, 'nesterov': self.args.nesterov}

        kwargs['lr'] = self.args.lr if lr is None else lr
        kwargs['weight_decay'] = self.args.weight_decay
        # embed()
        optimizer = optimizer_function(trainable, **kwargs)

        if self.args.load != '' and ckp is not None:
            print('Loading the optimizer from the checkpoint...')
            optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )

        return optimizer
    def make_loss_all(self, Loss):
        self.loss_list = [Loss(self.args, self.ckp) for _ in self.fractions]

    def make_scheduler_all(self, resume=-1, last_epoch=-1, reschedule=-1):
        ret = []
        for s in self.optimizer_list:
            ret.append(self.make_scheduler(s, resume, last_epoch, reschedule))
        self.scheduler_list = ret

    def make_scheduler(self, target, resume=-1, last_epoch=-1, reschedule=0):
        if self.args.decay.find('step') >= 0:
            milestones = list(map(lambda x: int(x), self.args.decay.split('-')[1:]))
            kwargs = {'milestones': milestones, 'gamma': self.args.gamma}

            scheduler_function = lrs.MultiStepLR
            # embed()
            kwargs['last_epoch'] = last_epoch
            scheduler = scheduler_function(target, **kwargs)

        if self.args.load != '' and resume > 0:
            for _ in range(resume): scheduler.step()
        if reschedule>0:
            for _ in range(reschedule): scheduler.step()
        return scheduler
    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()
            return x

        return [_prepare(a) for a in args]


class AverageMeter(object):
    """Record metrics information"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class Neuron_Hook:
    def __init__(self):
        self.hooks = []

    def subscribe(self, model, target: list, hook_fn: callable):
        for name, module in model.named_modules():
            if module.__class__.__name__.lower() in target:
                module.layer_name = name
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)

    def unsubscribe(self):
        for hook in self.hooks:
            hook.remove()


def test_firerate(net, img, target=['decomblock']):
        feature_maps = {}
        snnhook = Neuron_Hook()
        # @staticmethod
        def hook_fn(module, input, output):
            """Hook function to store intermediate layer outputs"""
            Nspks_max = 1  # make sure the [0,1] input
            num = input[0].unique()
            if len(num) <= Nspks_max+1 and input[0].max() <= Nspks_max and input[0].min() >= 0:
                layer_name = module.layer_name
                if layer_name not in feature_maps:
                    feature_maps[layer_name] = (input[0].sum() / input[0].numel()).item()
        net.eval()
        snnhook.subscribe(model=net, target=target, hook_fn=hook_fn)

        functional.reset_net(net)
        with torch.no_grad():
            functional.reset_net(net)
            net(img)

        fr_list = torch.tensor(list(feature_maps.values())).cpu()

        for i, key in enumerate(feature_maps.keys()):
            feature_maps[key] = fr_list[i].tolist()
        fr_list = copy.deepcopy(feature_maps)
        feature_maps.clear()
        snnhook.unsubscribe()
        max_fr_layer = list(max(fr_list.items(), key=lambda x: x[1]))
        return max_fr_layer