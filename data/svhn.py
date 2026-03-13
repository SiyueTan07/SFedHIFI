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


class SVHN_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, transform=None, target_transform=None, download=False, split = 'train'):
        self.root = root
        self.dataidxs = dataidxs
        self.train = split
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.split = self.train
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = datasets.SVHN(self.root, self.split, self.transform, self.target_transform, self.download)
        if self.train =='train':
            data = cifar_dataobj.data
            data = data.transpose((0, 2, 3, 1))
            target = np.array(cifar_dataobj.labels)
        else:
            data = cifar_dataobj.data
            data = data.transpose((0, 2, 3, 1))
            target = np.array(cifar_dataobj.labels)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = Image.fromarray(self.data[index]), self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def  gen_data_split(data, num_users, num_classes, class_partitions):
    N = data.shape[0]
    data_class_idx = {i: np.where(data== i)[0] for i in range(num_classes)}
    images_count_per_class = {i:len(data_class_idx[i]) for i in range(num_classes)}
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




def load_SVHN_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = SVHN_truncated(datadir, download=True, transform=transform, split='train')
    cifar10_test_ds = SVHN_truncated(datadir, download=True, transform=transform,split='test')
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test, cifar10_train_ds)




def partition_data(args):
    y_train = load_SVHN_data(args.dir_data)[1]
    num_classes = 10
    classes_per_user = 10 if args.split=='iid' else 3
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

def dirichlet_partition_data(args):
    alpha = args.alpha
    num_users = args.n_agents


    labels = load_SVHN_data(args.dir_data)[1]
    n_classes = labels.max() + 1  # 10
    idx = [np.argwhere(np.array(labels) == y).flatten() for y in range(n_classes)]
    label_distribution = np.random.dirichlet([alpha]*num_users, n_classes)

    agent_dataid = {i: np.array([], dtype='int64') for i in range(num_users)}
    for c, fracs in zip(idx, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            agent_dataid[i] = np.concatenate((agent_dataid[i], idcs), axis=0)

    return agent_dataid

# 画图
def dataset_stats(dict_users, dataset, args):
    num_classes = 10
    stats = {i: np.array([], dtype='int64') for i in range(len(dict_users))}
    for key, value in dict_users.items():
        for x in value:
            stats[key] = np.concatenate((stats[key], np.array([dataset[x][1]])), axis=0)  #dataset[x][1]]是数据x的标签

    nparray = np.zeros([num_classes, args.n_agents], dtype=int)
    for j in range(args.n_agents):
        cls = stats[j]
        cls_counter = Counter(cls)
        for i in range(num_classes):
            nparray[i][j] = cls_counter[i]

    fig, ax = plt.subplots()
    bottom = np.zeros([args.n_agents], dtype=int)
    for cls in range(num_classes):
        ax.bar(range(args.n_agents), nparray[cls], bottom=bottom, label='class{}'.format(cls))
        bottom += nparray[cls]
    ax.legend(loc='lower right')
    plt.title('Data Distribution')
    plt.xlabel('Clients')
    plt.ylabel('Amount of Training Data')
    plt.savefig('figs/fenbu_svhn_0.1.png', dpi=500)
    # plt.show()

def get_agent_loader(args, kwargs):
    loaders_train = []
    if args.split == 'iid':
        agent_dataid = partition_data(args)
    else:
        agent_dataid = dirichlet_partition_data(args)

    data_train = load_SVHN_data(args.dir_data)[4]

    dataset_stats(agent_dataid, data_train, args)



    norm_mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    norm_std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    g = torch.Generator()
    g.manual_seed(0)
    if not args.test_only:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:
            transform_list.insert(1, transforms.RandomHorizontalFlip())

        transform_train = transforms.Compose(transform_list)

    for i in range(args.n_agents):
        train_ds = SVHN_truncated(root=args.dir_data, dataidxs=agent_dataid[i], transform=transform_train, download=True, split='train')
        print(train_ds)

        train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker,generator=g, **kwargs)
        loaders_train.append(train_dl)

    return loaders_train

def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def get_loader(args, kwargs):
    norm_mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    norm_std=[x/255.0 for x in [63.0, 62.1, 66.7]]

    loader_train = None


    g = torch.Generator()
    g.manual_seed(0)
    if not args.test_only:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:
            transform_list.insert(1, transforms.RandomHorizontalFlip())

        transform_train = transforms.Compose(transform_list)
        loader_train = DataLoader(
            datasets.SVHN(
                root=args.dir_data,
                split='train',
                download=True,
                transform=transform_train),
            batch_size=args.batch_size, shuffle=True,worker_init_fn=seed_worker,generator=g, **kwargs
        )


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    loader_test = DataLoader(
        datasets.SVHN(
            root=args.dir_data,
            split='test',
            download=True,
            transform=transform_test),
        batch_size=250, shuffle=False, **kwargs
    )

    return loader_train, loader_test

