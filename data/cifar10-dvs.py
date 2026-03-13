from email import generator
from importlib import import_module
import torch.utils.data as data
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import torch
import numpy as np
import collections
from PIL import Image

from collections import Counter
import matplotlib.pyplot as plt

import os
from torch.utils.data import Dataset, ConcatDataset

# import seaborn as sns
# import sys

class DVSData(Dataset):
    # This code is form https://github.com/Gus-Lab/temporal_efficient_training
    def __init__(self, root, train=True, transform=None, target_transform=None, train_set_ratio=1.0, shape=32):
        # self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train_set_ratio = train_set_ratio
        self.train = train
        self.resize = transforms.Resize(size=(shape, shape))
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()
        if train:
            self.root = os.path.expanduser(root + '/train')
        else:
            self.root = os.path.expanduser(root + '/test')

        class_list = sorted(os.listdir(self.root))

        self.data_path_list = []
        self.label_list = []
        for i, class_name in enumerate(class_list):
            class_path = os.path.join(self.root, class_name)
            file_list = sorted(os.listdir(class_path))
            file_range = int(self.train_set_ratio * len(file_list))
            for files_name in file_list[: file_range]:
                self.data_path_list.append(os.path.join(class_path, files_name))
                self.label_list.append(float(i))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # data, target = torch.load(self.root + '/{}_np.pt'.format(index))
        data = torch.load(self.data_path_list[index], weights_only=False)
        target = self.label_list[index]

        if self.transform is not None:
            data = self.dvs_trans(data)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, torch.tensor(target).long()

    def dvs_trans(self, dvs_img):
        transformed_dvs_img = []
        for t in range(dvs_img.size(0)):
            data = self.imgx(dvs_img[t, ...])
            transformed_dvs_img.append(self.transform(data))
        dvs_img = torch.stack(transformed_dvs_img, dim=0)

        if self.train:
            flip = random.random() > 0.5
            if flip:
                dvs_img = torch.flip(dvs_img, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            dvs_img = torch.roll(dvs_img, shifts=(off1, off2), dims=(2, 3))
        return dvs_img

    def __len__(self):
        return len(self.data_path_list)


class CIFAR10DVS_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                 train_set_ratio=1.0, shape=32):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.train_set_ratio = train_set_ratio
        self.shape = shape

        self.base_dataset = DVSData(root, train, transform=self.transform, target_transform=self.target_transform,
                                    train_set_ratio=train_set_ratio, shape=shape)

        if dataidxs is not None:
            self.data_path_list = [self.base_dataset.data_path_list[i] for i in dataidxs]
            self.label_list = [self.base_dataset.label_list[i] for i in dataidxs]
        else:
            self.data_path_list = self.base_dataset.data_path_list
            self.label_list = self.base_dataset.label_list

    def __getitem__(self, index):
        data, target = self.base_dataset.__getitem__(index)
        return data, target

    def __len__(self):
        return len(self.data_path_list)


def load_cifar10dvs_data(datadir, shape=32):
    cifar10dvs_train_ds = CIFAR10DVS_truncated(datadir, train=True, shape=shape)
    cifar10dvs_test_ds = CIFAR10DVS_truncated(datadir, train=False, shape=shape)

    X_train_paths = cifar10dvs_train_ds.data_path_list
    y_train = cifar10dvs_train_ds.label_list
    X_test_paths = cifar10dvs_test_ds.data_path_list
    y_test = cifar10dvs_test_ds.label_list

    return (X_train_paths, y_train, X_test_paths, y_test, cifar10dvs_train_ds)


def gen_data_split(labels, num_users, num_classes, class_partitions):
    N = len(labels)
    data_class_idx = {i: np.where(np.array(labels) == i)[0] for i in range(num_classes)}
    images_count_per_class = {i: len(data_class_idx[i]) for i in range(num_classes)}

    for data_idx in data_class_idx:
        np.random.shuffle(data_class_idx[data_idx])

    user_data_idx = collections.defaultdict(list)
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(images_count_per_class[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    for usr in user_data_idx:
        np.random.shuffle(user_data_idx[usr])

    return user_data_idx


def partition_data(args, shape=32):
    _, y_train, _, _, _ = load_cifar10dvs_data(args.dir_data, shape=shape)
    num_classes = 10
    classes_per_user = 10 if args.split == 'iid' else 3
    num_users = args.n_agents

    assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
    count_per_class = (classes_per_user * num_users) // num_classes

    class_dict = {}
    for i in range(num_classes):
        probs = np.random.uniform(1, 1, size=count_per_class)
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    class_partitions = collections.defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

    agent_dataid = gen_data_split(y_train, num_users, num_classes, class_partitions)
    return agent_dataid


def dirichlet_partition_data(args, shape=32):
    alpha = args.alpha
    num_users = args.n_agents

    _, y_train, _, _, _ = load_cifar10dvs_data(args.dir_data, shape=shape)
    n_classes = int(max(y_train) + 1)

    idx = [np.argwhere(np.array(y_train) == y).flatten() for y in range(n_classes)]

    label_distribution = np.random.dirichlet([alpha] * num_users, n_classes)

    agent_dataid = {i: np.array([], dtype='int64') for i in range(num_users)}
    for c, fracs in zip(idx, label_distribution):
        splits = (np.cumsum(fracs)[:-1] * len(c)).astype(int)
        for i, idcs in enumerate(np.split(c, splits)):
            agent_dataid[i] = np.concatenate((agent_dataid[i], idcs), axis=0)

    return agent_dataid


def dataset_stats(dict_users, dataset, args):
    num_classes = 10
    stats = {i: [] for i in range(len(dict_users))}

    for key, value in dict_users.items():
        for x in value:
            _, label = dataset[x]  # ��ȡ��ǩ
            stats[key].append(label)

    nparray = np.zeros([num_classes, args.n_agents], dtype=int)
    for j in range(args.n_agents):
        cls_counter = Counter(stats[j])
        for i in range(num_classes):
            nparray[i][j] = cls_counter.get(i, 0)

    fig, ax = plt.subplots()
    bottom = np.zeros([args.n_agents], dtype=int)
    for cls in range(num_classes):
        ax.bar(range(args.n_agents), nparray[cls], bottom=bottom, label=f'class{cls}')
        bottom += nparray[cls]

    ax.legend(loc='lower right')
    plt.title('CIFAR10-DVS Data Distribution')
    plt.xlabel('Clients')
    plt.ylabel('Amount of Training Data')
    plt.savefig('figs/cifar10dvs_data_distribution.png', dpi=500)
    plt.close()


def get_agent_loader(args, kwargs, shape=32):
    loaders_train = []

    if args.split == 'iid':
        agent_dataid = partition_data(args, shape=shape)
    else:
        agent_dataid = dirichlet_partition_data(args, shape=shape)

    data_train = load_cifar10dvs_data(args.dir_data, shape=shape)[4]

    g = torch.Generator()
    g.manual_seed(0)

    for i in range(args.n_agents):
        train_ds = CIFAR10DVS_truncated(
            root=args.dir_data,
            dataidxs=agent_dataid[i],
            train=True,
            shape=shape
        )

        train_dl = DataLoader(
            dataset=train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            **kwargs
        )
        loaders_train.append(train_dl)

    return loaders_train


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(args, kwargs, shape=32):
    g = torch.Generator()
    g.manual_seed(0)

    loader_train = None
    if not args.test_only:
        # ����ѵ����������
        train_dataset = CIFAR10DVS_truncated(
            args.dir_data,
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(shape, padding=4),
                transforms.RandomHorizontalFlip(),
            ]),
            shape=shape
        )

        loader_train = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            **kwargs
        )

    test_dataset = CIFAR10DVS_truncated(
        args.dir_data,
        train=False,
        transform=None,
        shape=shape
    )

    loader_test = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        **kwargs
    )

    return loader_train, loader_test

