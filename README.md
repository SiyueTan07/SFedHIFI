## SFedHIFI: Fire Rate-Based Heterogeneous Information Fusion for Spiking Federated Learning (AAAI'2026)

## Abstract
Spiking Federated Learning (SFL) has been widely studied with the energy efficiency of Spiking Neural Networks (SNNs). However, existing SFL methods require model homogeneity and assume all clients have sufficient computational resources, resulting in the exclusion of some resource-constrained clients.  To address the prevalent system heterogeneity in real-world scenarios, enabling heterogeneous SFL systems that allow clients to adaptively deploy models of different scales based on their local resources is crucial. To this end, we introduce SFedHIFI, a novel **S**piking **Fed**erated Learning framework with Fire Rate-Based **H**eterogeneous **I**nformation **F**us**i**on. Specifically, we leverage channel pruning to deploy a series of SNN models with adjustable complexity on clients with varying computational resources. By channel-wise matrix decomposition and the proposed heterogeneous information fusion module, SFedHIFI enables cross-scale aggregation across client models of different widths, thereby facilitating the learning from more comprehensive local datasets. Extensive experiments on three public benchmarks demonstrate that SFedHIFI can effectively enable heterogeneous SFL, consistently outperforming all three baseline methods. Compared with ANN-based FL, it achieves significant energy savings with only a marginal trade-off in accuracy.

## Overview

- **SFedHIFI Framework**

<img src="Pictures/overview_01.png" alt="overview_01" style="zoom:40%;" />

## Getting Started

### 1. Requirements

Install the requirements using a `conda` environment:
```
conda env create -f environment.yml
```
(Optional) We use [SwanLab](https://github.com/SwanHubX/SwanLab) to log experiment data. Please refer to the official documentation and configure the API
``
swanlab.login(api_key='')
``
 in [main_FL.py](main_FL.py).

### 2. Train SFedHIFI

The script can be found under [demo.sh](demo.sh), below is an example of SFedHIFI on CIFAR-10 dataset with IID set.

```python
CUDA_VISIBLE_DEVICES=0 python main_FL.py --n_agents 10 --dir_data /home/user/dataset --data_train cifar10 --data_test cifar10 --n_joined 5 --model_type snn --tucker --tucker_epochs 225 --split iid --T 10 --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --template ResNet18 --model spiking_resnet_flanc --basis_fraction 0.125 --n_basis 0.25
```

### 3. Evaluation with baseline.

The script can be found under [demo.sh](demo.sh), below is an example of FedAVG on CIFAR-10 dataset with IID set.

```python
CUDA_VISIBLE_DEVICES=0 python main_FL.py --n_agents 10 --dir_data /home/user/Dataset --classic fedavg --data_train cifar10 --data_test cifar10 --n_joined 5 --model_type snn --split iid --T 10 --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --template ResNet18 --model spiking_resnet_flanc --basis_fraction 0.125 --n_basis 0.25
```


## Acknowledgements
This code is built on [Spikingjelly](https://github.com/fangwei123456/spikingjelly) and [FLANC](https://github.com/HarukiYqM/All-In-One-Neural-Composition). We thank the authors for sharing their codes.