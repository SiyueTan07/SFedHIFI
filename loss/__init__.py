import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class TET_loss(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss(), means=1.0, lamb=1e-3)  -> Tensor:
        super().__init__()
        """
        calculate TET loss.
        $L_{TET} = \frac{1}{T} \cdot \sum_{t=1}^{T} L_{CE}[O(t), y]$
        This code is form https://github.com/Gus-Lab/temporal_efficient_training
        :param outputs: the predict results
        :param labels: the true labels
        :param criterion: the criterion in TET loss
        :param means:  the target of regularization loss.
        :param lamb: the ratio of regularization loss, which aims to avoid the outputs occur outliers.
        :return:
        """
        self.crt = criterion
        self.means = means
        self.lamb = lamb

    def forward(self, outputs, labels):
        loss_es = 0
        T = outputs.size(0)
        for t in range(T):
            loss_es += self.crt(outputs[t, ...], labels)
        loss_es = loss_es / T  # L_TET
        if self.lamb != 0:
            MSE_loss = torch.nn.MSELoss()
            y = torch.zeros_like(outputs).fill_(self.means)
            regularization_loss = MSE_loss(outputs, y)  # L_mse
        else:
            regularization_loss = 0
        return (1 - self.lamb) * loss_es + self.lamb * regularization_loss

class SNN_loss(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss())  -> Tensor:
        super().__init__()
        self.crt = criterion

    def forward(self, outputs, labels):
        outputs = outputs.mean(0)
        loss = self.crt(outputs, labels)
        return loss

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function...')
        self.args = args

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()

        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'NLL':
                loss_function = nn.NLLLoss() if args.model_type.lower() == 'ann' else SNN_loss(criterion=nn.NLLLoss())
            elif loss_type == 'CE':
                loss_function = nn.CrossEntropyLoss() if args.model_type.lower() == 'ann' else SNN_loss(criterion=nn.CrossEntropyLoss())
            elif loss_type == 'MSE':
                loss_function = nn.MSELoss() if args.model_type.lower() == 'ann' else SNN_loss(criterion=nn.MSELoss())
            elif loss_type == 'TET':
                loss_function = TET_loss()

        self.loss.append({
            'type': loss_type,
            'weight': float(weight),
            'function': loss_function
        })

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        print('Loss function:')
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log_train = torch.Tensor()
        self.log_test = torch.Tensor()

        device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, prediction, label, train=True):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](prediction, label)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

        loss_sum = sum(losses)
        topk = self.accuracy(prediction, label)

        if train:
            log = self.log_train
        else:
            log = self.log_test

        log[-1, 0] += loss_sum.item() * prediction.size(0)
        log[-1, 1] += topk[0]
        log[-1, 2] += topk[1]

        return loss_sum, topk

    def accuracy(self, prediction, label):
        if self.args.model_type.lower() == 'snn':
            prediction = prediction.mean(0)
        topk = (1, 5)
        _, pred = prediction.topk(max(topk), 1, largest=True, sorted=True)
        correct = pred.eq(label.unsqueeze(-1))
        res = []
        for k in topk:
            correct_k = correct[:, :k].float().sum()
            res.append(100.0 * (prediction.size(0) - correct_k.item()))

        return res

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'): l.scheduler.step()

    def start_log(self, train=True):
        empty_log = torch.zeros(1, 3)
        if train:
            self.log_train = torch.cat((self.log_train, empty_log))
        else:
            self.log_test = torch.cat((self.log_test, empty_log))

    def end_log(self, n_samples, train=True):
        if train:
            self.log_train[-1].div_(n_samples)
        else:
            self.log_test[-1].div_(n_samples)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath):

        splits = ['Test']
        logs = [self.log_test]
        if not self.args.test_only:
            if self.log_train.shape[0] > 0:
                splits.append('Training')
                logs.append(self.log_train)

        for s, (split, log) in enumerate(zip(splits, logs)):
            if s == 0:
                if self.log_train.shape[0] < self.log_test.shape[0]:
                    axis = np.array(list(range(0, len(self.log_test))))
                else:
                    axis = np.array(list(range(1, len(self.log_test) + 1)))
            else:
                axis = np.array(list(range(1, len(self.log_train) + 1)))
            for i, measure in enumerate(('NLL', 'Top-1', 'Top-5')):
                label = '{} ({})'.format(measure, split)
                fig = plt.figure()
                plt.title(label)

                best = log[:, i].min()
                plt.plot(
                    axis,
                    log[:, i].numpy(),
                    label='Best: {:.4f}'.format(best)
                )
                plt.legend()
                plt.xlabel('Epochs')
                if measure == 'NLL': 
                    plt.ylabel('Loss')
                else:
                    plt.ylabel('Error (%)')
                plt.grid(True)
                plt.savefig('{}/{}_{}.pdf'.format(apath, measure, split))
                plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log_train, os.path.join(apath, 'train_log.pt'))
        torch.save(self.log_test, os.path.join(apath, 'test_log.pt'))

    def load(self, apath, cpu=False):
        kwargs = {}
        if cpu: kwargs = {'map_location': lambda storage, loc: storage}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log_train = torch.load(os.path.join(apath, 'train_log.pt'))
        self.log_test = torch.load(os.path.join(apath, 'test_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

