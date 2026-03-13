#!bin/bash

# SFedHIFI on Fasion-MNIST with iid
python main_FL.py --n_agents 10 --dir_data /home/user/dataset  --data_train fashion-mnist --data_test fashion-mnist --n_joined 5 --model_type snn --tucker --tucker_epochs 225 --lambdaR 0.1 --split iid --T 10 --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.01 --fraction_list 0.25,0.5,0.75,1 --template ResNet18 --model spiking_cnn_flanc --basis_fraction 0.125 --n_basis 0.25

# SFedHIFI on CIFAR-10 with iid
python main_FL.py --n_agents 10 --dir_data /home/user/dataset --data_train cifar10 --data_test cifar10 --n_joined 5 --model_type snn --tucker --tucker_epochs 225 --split iid --T 10 --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --template ResNet18 --model spiking_resnet_flanc --basis_fraction 0.125 --n_basis 0.25

# SFedHIFI on CIFAR-10 with iid
python main_FL.py --n_agents 10 --dir_data /home/user/dataset --data_train cifar100 --data_test cifar100 --n_joined 5 --model_type snn --tucker --tucker_epochs 360 --split iid --T 10 --local_epochs 2 --batch_size 32 --epochs 800 --decay step-300-575 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --template ResNet18 --model spiking_resnet_flanc --basis_fraction 0.125 --n_basis 0.25

BASELINE=[fedavg, fedprox, fednova]
# Baselines on Fasion-MNIST with iid
python main_FL.py --n_agents 10 --dir_data /home/user/Dataset --classic [BASELINE] --data_train fashion-mnist --data_test fashion-mnist --n_joined 5 --model_type snn --lambdaR 0.1 --split iid --T 10 --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.01 --fraction_list 0.25,0.5,0.75,1 --template ResNet18 --model spiking_cnn_flanc --basis_fraction 0.125 --n_basis 0.25

# Baselines on CIFAR-10 with iid
python main_FL.py --n_agents 10 --dir_data /home/user/Dataset --classic [BASELINE] --data_train cifar10 --data_test cifar10 --n_joined 5 --model_type snn --split iid --T 10 --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --template ResNet18 --model spiking_resnet_flanc --basis_fraction 0.125 --n_basis 0.25

# Baselines on CIFAR-100 with iid
python main_FL.py --n_agents 10 --dir_data /home/user/Dataset --classic [BASELINE] --data_train cifar100 --data_test cifar100 --n_joined 5 --model_type snn --split iid --T 10 --local_epochs 2 --batch_size 32 --epochs 800 --decay step-300-575 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --template ResNet18 --model spiking_resnet_flanc --basis_fraction 0.125 --n_basis 0.25

# For Non-IID partition set, only need set [--split] to 'noiid' and set [--alpha].
