
import torchvision.models as models
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model import common
import torch
import torch.nn.functional as F
import copy
from spikingjelly.activation_based import layer, neuron, functional


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return layer.SeqToANNContainer(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size // 2), groups=groups, bias=bias, dilation=dilation))

def conv1x1(in_planes, out_planes, kernel_size=1, stride=1,bias=False):
    """1x1 convolution"""
    return layer.SeqToANNContainer(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=bias))

def make_model(args, parent=False):

    return CNN_FLANC(args)

class conv_basis(nn.Module):
    def __init__(self, filter_bank, in_channels, basis_size, n_basis, kernel_size, stride=1, bias=True):
        super(conv_basis, self).__init__()
        self.in_channels = in_channels
        self.n_basis = n_basis
        self.kernel_size = kernel_size
        self.basis_size = basis_size
        self.stride = stride
        self.group = in_channels // basis_size
        self.weight = filter_bank
        self.bias = nn.Parameter(torch.zeros(n_basis)) if bias else None
        # self.sn = neuron.IFNode()
        functional.set_step_mode(self, step_mode='m')
        #print(stride)
    def forward(self, x):
        if self.group == 1:
            conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=self.bias, stride=self.stride,
                                  padding=self.kernel_size//2)
            conv2d.weight = self.weight
            conv2d.padding = self.kernel_size//2
            conv2d.padding_mode = 'zeros'

            x = layer.SeqToANNContainer(conv2d)(x)
        else:

            x_tlist = []
            for t in range(x.shape[0]):
                x_list = []
                xt = x[t]
                for xi in torch.split(xt, self.basis_size, dim=1):
                    conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=self.bias, stride=self.stride,
                                          padding=self.kernel_size // 2)
                    conv2d.weight = self.weight
                    conv2d.padding = self.kernel_size // 2
                    conv2d.padding_mode = 'zeros'
                    x1 = conv2d(xi)
                    # print(1 conv1, x1.shape)
                    x_list.append(x1)
                x_tlist.append(torch.cat(x_list, dim=1))
            x = torch.stack(x_tlist, dim=0)
        return x

    def __repr__(self):
        s = 'Conv_basis(in_channels={}, basis_size={}, group={}, n_basis={}, kernel_size={}, out_channel={})'.format(
            self.in_channels, self.basis_size, self.group, self.n_basis, self.kernel_size, self.group * self.n_basis)
        return s

class DecomBlock(nn.Module):
    def __init__(self, filter_bank, in_channels, out_channels, n_basis, basis_size, kernel_size,
                 stride=1, bias=False, conv=conv3x3, norm=common.default_norm, act=common.default_act):
        super(DecomBlock, self).__init__()
        group = in_channels // basis_size
        modules = [conv_basis(filter_bank,in_channels, basis_size, n_basis, kernel_size, stride, bias)]
        modules.append(neuron.IFNode())
        modules.append(conv(group * n_basis, out_channels, kernel_size=1, stride=1, bias=bias))
        self.conv = nn.Sequential(*modules)
        functional.set_step_mode(self, step_mode='m')
    def forward(self, x):
        return self.conv(x)


class CNN_FLANC(nn.Module):

    """ Simple network"""

    # def __init__(self, args, T = 10, spiking_neuron: callable = None, **kwargs):
    def __init__(self,  args, spiking_neuron: callable = None, **kwargs):
        super(CNN_FLANC, self).__init__()

        self.T = args.T
        self.th = 0.5
        basis_fract = args.basis_fraction
        net_fract= args.net_fraction
        n_basis = args.n_basis

        self.head = layer.SeqToANNContainer(nn.Conv2d(1, 64, 3, stride=1, padding=1))
        m1 = round(128*n_basis)
        n1 = round(64*basis_fract)
        self.filter_bank_1 = nn.Parameter(torch.empty(m1, n1, 3, 3))

        m2 = round(128*n_basis)
        n2 = round(128*basis_fract)
        self.filter_bank_2 = nn.Parameter(torch.empty(m2, n2, 3, 3))

        X1 = torch.empty(m1, n1, 3, 3)
        torch.nn.init.orthogonal(X1)
        self.filter_bank_1.data = copy.deepcopy(X1)
        X2 = torch.empty(m2, n2, 3, 3)
        torch.nn.init.orthogonal(X2)
        self.filter_bank_2.data = copy.deepcopy(X2)
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        out_1 = round(128*net_fract)
        self.conv1 = DecomBlock(self.filter_bank_1, 64, out_1, m1, n1, kernel_size=3, bias=False) # 28
        self.sn2 = neuron.IFNode(v_threshold=self.th)
        # self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2)) # 14

        out_2 = round(128*net_fract)
        self.conv2 = DecomBlock(self.filter_bank_2, out_1, out_2, m2, n2, kernel_size=3, bias=False)
        self.sn3 = neuron.IFNode(v_threshold=self.th)
        # self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2)) # 7

        # self.classifier = layer.Linear(out_2 * 7 * 7*32, 10)
        self.classifier = layer.SeqToANNContainer(nn.Linear(out_2 * 7 * 7, 10))

        self.sn4 = neuron.IFNode(v_threshold=self.th)

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.head(x)    # (10,32,1,28,28)
        x = self.sn1(x)
        x = self.conv1(x)   # (10,32,64,28,28)
        x = self.sn2(x)     # (10,32,128,28,28)
        x = self.pool1(x)   # (10,32,128,28,28)

        x = self.conv2(x)   # (10,32,128,14,14)
        x = self.sn3(x)     # (10,32,128,14,14)
        x = self.pool2(x)   # (10,32,128,14,14)
        x = torch.flatten(x, 2) # (10,32,128,7,7)
        x = self.classifier(x)  # (10,32,6272)

        x = self.sn4(x)       # (10,32,10)
        return x

def loss_type(loss_para_type):
    if loss_para_type == 'L1':
        loss_fun = nn.L1Loss()
    elif loss_para_type == 'L2':
        loss_fun = nn.MSELoss()
    else:
        raise NotImplementedError
    return loss_fun

def orth_loss(model, args, para_loss_type='L2'):

    loss_fun = loss_type(para_loss_type)

    loss = 0
    for l_id in range(1,3):
        filter_bank = getattr(model,"filter_bank_"+str(l_id))

        all_bank = filter_bank
        num_all_bank = filter_bank.shape[0]
        B = all_bank.view(num_all_bank, -1)
        D = torch.mm(B,torch.t(B))
        D = loss_fun(D, torch.eye(num_all_bank, num_all_bank).cuda())
        loss = loss + D
    return loss


