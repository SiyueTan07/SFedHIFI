import argparse
import template
import numpy as np

parser = argparse.ArgumentParser(description='Deep Kernel Clustering')


parser.add_argument('--T', type=int, default=10, help='simulating time-steps')
parser.add_argument('--alpha', type=float, default=0.1, help='dirichlet alpha')
parser.add_argument('--tucker', action='store_true',
                    help='weather use tucker decomposition for anchors [FedFGA].')
parser.add_argument('--tucker_epochs', type=int, default=None, help='number of epochs for tucker aggregation.')
parser.add_argument('--merger_epoch', type=int, default=1, help='Tucker aggregation frequency: every [merger_epoch] epochs.')
parser.add_argument('--all_tucker', type=str, default=None,
                    help='use tucker decomposition for ablation')
parser.add_argument('--model_type', type=str, default='snn', help='[ann, snn] model type.')
parser.add_argument('--sn_type', type=str, default='lif', help='[LIF, IF] sn type.')
parser.add_argument('--classic', type=str, default=None, help='[fedavg, fedprox, fednova] classic FL method.')


# parser.add_argument('--gpus', nargs='+',default=None, type=str, help='gpus')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='',
                    help='You can set various templates in template.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='disable CUDA training')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

parser.add_argument('--gpus', nargs='+',default=None, type=str, help='gpus')

# Data specifications
parser.add_argument('--dir_data', default='/home/user/dataset/',
                    help='dataset directory')
parser.add_argument('--data_train', default='CIFAR10',
                    help='train dataset name')
parser.add_argument('--data_test', default='CIFAR10',
                    help='test dataset name')
parser.add_argument('--dvs', action='store_true',
                    help='use dvs datasets.')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_flip', action='store_true',
                    help='disable flip augmentation')
parser.add_argument('--crop', type=int, default=1,
                    help='enables crop evaluation')

# Model specifications
parser.add_argument('--model', default='DenseNet',
                    help='model name')
parser.add_argument('--vgg_type', type=str, default='16',
                    help='VGG type')
parser.add_argument('--download', action='store_true',
                    help='download pre-trained model')
parser.add_argument('--base', default='',
                    help='base model')
parser.add_argument('--base_p', default='',
                    help='base model for parent')
parser.add_argument('--channel', type=int, default=1,
                    help='in_channel for 4-layer CNN.')

parser.add_argument('--resume_from', type=str, default='')
parser.add_argument('--act', default='relu',
                    help='activation function')
parser.add_argument('--pretrained', default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', default='',
                    help='pre-trained model directory')

parser.add_argument('--depth', type=int, default=100,
                    help='number of convolution modules')
parser.add_argument('--in_channels', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--k', type=int, default=12,
                    help='DenseNet grownth rate')
parser.add_argument('--reduction', type=float, default=1,
                    help='DenseNet reduction rate')
parser.add_argument('--bottleneck', action='store_true',
                    help='ResNet/DenseNet bottleneck')

parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size')
parser.add_argument('--no_bias', action='store_true',
                    help='do not use bias term for conv layer')
parser.add_argument('--precision', default='single',
                    help='model and data precision')

parser.add_argument('--multi', type=str, default='full-256',
                    help='multi clustering')
parser.add_argument('--n_init', type=int, default=1,
                    help='number of differnt k-means initialization')
parser.add_argument('--max_iter', type=int, default=4500,
                    help='maximum iterations for kernel clustering')
parser.add_argument('--symmetry', type=str, default='i',
                    help='clustering algorithm')
parser.add_argument('--init_seeds', type=str, default='random',
                    help='kmeans initialization method')
parser.add_argument('--scale_type', type=str, default='kernel_norm_train',
                    help='scale parameter configurations')
parser.add_argument('--n_bits', type=int, default=16,
                    help='number of bits for scale parameters')
parser.add_argument('--top', type=int, default=1, choices=[1, -1],
                    help='save model for top1 or top5 error. top1: 1, top5: -1.')

# Group
parser.add_argument('--group_size', type=int, default=16,
                    help='group size for the network of filter group approximation, ECCV 2018 paper.')

# DenseNet Basis
parser.add_argument('--n_group', type=int, default=1,
                    help='number of groups for the compression of densenet')
parser.add_argument('--k_size1', type=int, default=3,
                    help='kernel size 1')
parser.add_argument('--k_size2', type=int, default=3,
                    help='kernel size 2')
parser.add_argument('--inverse_index', action='store_true',
                    help='index the basis using inverse index')
parser.add_argument('--transition_group', type=int, default=6,
                    help='number of groups in the transition layer of DenseNet')

# ResNet Basis
parser.add_argument('--basis_size1', type=int, default=16,
                    help='basis size for the first res group in ResNet')
parser.add_argument('--basis_size2', type=int, default=32,
                    help='basis size for the second res group in ResNet')
parser.add_argument('--basis_size3', type=int, default=64,
                    help='basis size for the third res group in ResNet')
parser.add_argument('--n_basis1', type=int, default=24,
                    help='number of basis for the first res group in ResNet')
parser.add_argument('--n_basis2', type=int, default=48,
                    help='number of basis for the second res group in ResNet')
parser.add_argument('--n_basis3', type=int, default=84,
                    help='number of basis for the third res group in ResNet')

# more model specification
parser.add_argument('--vgg_decom_type', type=str, default='all',
                    help='vgg decomposition type, valid value all, select')
parser.add_argument('--basis_size_str', type=str, default='',
                    help='basis size')
parser.add_argument('--n_basis_str', type=str, default='',
                    help='number of basis')
parser.add_argument('--basis_size', type=int, default=128,
                    help='basis size')
parser.add_argument('--n_basis', type=float, default=1,
                    help='number of basis')
parser.add_argument('--pre_train_optim', type=str, default='.',
                    help='pre-trained weights directory')
parser.add_argument('--unique_basis', action='store_true',
                    help='whether to use the same basis for the two convs in the Residual Block')
parser.add_argument('--loss_orth', action='store_true',
                    help='whether to use default loss_norm')
parser.add_argument('--split', type=str, default='iid',
                    help='whether to use default loss_norm')
# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--resume', type=int, default=-1,
                    help='load the model from the specified epoch')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training')

# Optimization specifications
parser.add_argument('--linear', type=int, default=1,
                    help='linear scaling rule')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate')
parser.add_argument('--decay', default='step-250-375',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor')

parser.add_argument('--optimizer', type=str, default='SGD',
                    help='optimizer to use')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='enable nesterov momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM betas')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay parameter')
parser.add_argument('--basis_fraction',type=float,default=0.5)
parser.add_argument('--net_fraction',type=float,default=1)
parser.add_argument('--fraction_list',type=str,default='')
parser.add_argument('--sync',action='store_true')
parser.add_argument('--n_agents',type=int, default=-1,
                    help="num of agents in whole federations")
parser.add_argument('--n_joined',type=int, default=-1,
                    help="num of agents joined in a federation")
parser.add_argument('--local_epochs',type=int, default=1,
                    help="local training epochs of agents")
# Loss specifications
parser.add_argument('--loss', default='1*TET',
                    help='loss function configuration')
parser.add_argument('--no_loss_orth', action='store_true',
                    help='do not use orthogonal term in loss')
parser.add_argument('--lambdaR',type=float, default=10,
                    help='orthogonal loss parameter')
# Log specifications
parser.add_argument('--dir_save', default='./experiment',
                    help='the directory used to save')
parser.add_argument('--save', default='test',
                    help='file name to save')
parser.add_argument('--load', default='',
                    help='file name to load')
parser.add_argument('--print_every', type=int, default=100,
                    help='print intermediate status per N batches')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--compare', type=str, default='',
                    help='experiments to compare with')
parser.add_argument('--project', type=str, default='')

# ============================================================
# SIFS / SATR extensions on top of SFedHIFI FLANC
# ------------------------------------------------------------
# --sifs enables element-wise dynamic sparse training.
# --satr enables conservative component-wise filter-bank selection.
# When neither flag is given, the default SFedHIFI / Tucker behaviour is used.
# ============================================================
parser.add_argument('--sifs', action='store_true',
                    help='Enable Spike-Induced Federated Sparsity (SIFS).')
parser.add_argument('--satr', action='store_true',
                    help='Enable Spike-Aware Tucker Rank/Component Calibration (SATR).')
parser.add_argument('--satr_retain_ratio', type=float, default=0.5,
                    help='Final retained fraction of filter-bank components for SATR.')
parser.add_argument('--satr_init_retain_ratio', type=float, default=1.0,
                    help='Initial retained component fraction for SATR before annealing.')
parser.add_argument('--satr_warmup_rounds', type=int, default=5,
                    help='Number of rounds before SATR starts reducing component budget.')
parser.add_argument('--satr_final_round', type=int, default=None,
                    help='Round at which SATR reaches target retain ratio (defaults to int(0.75*epochs)).')
parser.add_argument('--satr_update_interval', type=int, default=1,
                    help='Update SATR component masks every K rounds after warmup.')
parser.add_argument('--satr_baseline', type=str, default='full',
                    choices=['full', 'static', 'random', 'magnitude', 'taylor', 'spike', 'no_norm', 'no-normalisation', 'no-normalization'],
                    help='SATR scoring mode / baseline. static keeps all components and reduces to SFedHIFI sync.')
parser.add_argument('--satr_silence_threshold', type=float, default=1e-3,
                    help='Mean spike rate below which a channel is treated as silent for SATR.')
parser.add_argument('--satr_spike_weight', type=float, default=1.0,
                    help='Weight of the normalised spike-utilisation component in SATR.')
parser.add_argument('--satr_taylor_weight', type=float, default=1.0,
                    help='Weight of the normalised Taylor component in SATR.')
parser.add_argument('--satr_log_scores', action='store_true',
                    help='Log SATR live component ratios and score statistics to swanlab.')
parser.add_argument('--satr_plot_interval', type=int, default=0,
                    help='Save SATR heatmaps every K rounds when --satr_log_scores is enabled. Default 0 disables plot files; .pt diagnostics are still saved.')
parser.add_argument('--sifs_target_sparsity', type=float, default=0.5,
                    help='Final target sparsity over filter-bank parameters.')
parser.add_argument('--sifs_init_sparsity', type=float, default=0.0,
                    help='Initial sparsity at round 0 (s_0).')
parser.add_argument('--sifs_warmup_rounds', type=int, default=5,
                    help='Number of rounds before the first prune/grow update.')
parser.add_argument('--sifs_final_round', type=int, default=None,
                    help='Round at which sparsity should reach target (defaults to int(0.75*epochs)).')
parser.add_argument('--sifs_mask_update_interval', type=int, default=2,
                    help='Apply a prune/grow update every K rounds after warmup.')
parser.add_argument('--sifs_rebirth_ratio', type=float, default=0.3,
                    help='Fraction of currently-zero weights considered as grow candidates per update.')
parser.add_argument('--sifs_silence_threshold', type=float, default=1e-3,
                    help='Mean spike rate below which a presynaptic channel is considered silent.')
parser.add_argument('--sifs_taylor_weight', type=float, default=1.0,
                    help='Weight of the Taylor magnitude term in I_Prune (tie breaker among silent connections).')
parser.add_argument('--sifs_silence_weight', type=float, default=10.0,
                    help='Weight of the spike-emptiness term in I_Prune (dominant term).')
parser.add_argument('--sifs_crisis_weight', type=float, default=0.0,
                    help='Coefficient lambda for the spike-rate crisis regularizer.')
parser.add_argument('--sifs_crisis_floor', type=float, default=0.02,
                    help='Minimum mean spike rate r_min used in the crisis regularizer.')
parser.add_argument('--sifs_aggregator', type=str, default='mask_aware',
                    choices=['mask_aware', 'fedavg'],
                    help='Aggregator type when SIFS is on. mask_aware = divide by per-element coverage; fedavg = SFedHIFI default.')
parser.add_argument('--sifs_log_silence', action='store_true',
                    help='Log per-layer spike-emptiness statistics to swanlab.')

args = parser.parse_args()
template.set_template(args)

if args.epochs == 0:
    args.epochs = 1e8

if args.pretrained and args.pretrained != 'download':
    args.n_init = 1
    args.max_iter = 1

if args.fraction_list:
    fracts = args.fraction_list.split(',')
    fracts = [float(f) for f in fracts]
    args.fraction_list = fracts

if args.tucker_epochs == None:
    args.tucker_epochs = int(np.round(0.45 * args.epochs))

if args.sifs and args.sifs_final_round is None:
    args.sifs_final_round = int(np.round(0.75 * args.epochs))

if args.satr and args.satr_final_round is None:
    args.satr_final_round = int(np.round(0.75 * args.epochs))
