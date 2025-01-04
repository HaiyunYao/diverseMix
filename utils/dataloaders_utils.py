import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from transformers import CLIPModel
from torchvision import datasets, transforms
import torchvision.transforms as transforms

from tqdm import tqdm
import config

_tokenizer = _Tokenizer()


def set_train_loader(args, subset=False, max_count=0):
    root = args.root_dir
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    batch_size = args.batch_size
    batch_size = 256
    shuffle = True
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'train')
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
    dataset = datasets.ImageFolder(path, transform=preprocess)
    if subset:
        from collections import defaultdict
        classwise_count = defaultdict(int)
        indices = []
        for i, label in enumerate(dataset.targets):
            if classwise_count[label] < max_count:
                indices.append(i)
                classwise_count[label] += 1
        dataset = torch.utils.data.Subset(dataset, indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return train_loader


def set_val_loader(args):
    root = args.root_dir
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'val')
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'val')
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
    dataset = datasets.ImageFolder(path, transform=preprocess)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return val_loader


def set_ood_loader_ImageNet(args, out_dataset):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    root = os.path.join(args.root_dir, 'ImageNet_OOD_dataset')
    # normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':  # filtered places
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Places'), transform=preprocess)
    elif out_dataset == 'placesbg':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'placesbg'), transform=preprocess)
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.test_batch_size, shuffle=False,
                                                num_workers=4)

    return testloaderOut


class RandomCrop(object):
    def __init__(self, n_crop=2):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        self.n_crop = n_crop
        self.random_crop = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        views = [self.random_crop(x).unsqueeze(dim=0) for _ in range(self.n_crop)]
        views = torch.cat(views, dim=0)
        return views


class RandomCropAndMask(object):
    def __init__(self, n_crop=2, n_mask=2):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        self.n_crop = n_crop
        self.n_mask = n_mask
        self.random_crop = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        # self.random_mask = transforms.Compose([
        #     transforms.Resize(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     # transforms.RandomErasing(p=1),
        #     transforms.RandomErasing(p=1, scale=(0.10, 0.50)),
        #     transforms.CenterCrop(224),
        #     normalize
        # ])
        self.random_mask = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=1),
            transforms.RandomErasing(p=1, scale=(0.10, 0.50)),
            normalize
        ])

    def __call__(self, x):
        views_crop = [self.random_crop(x).unsqueeze(dim=0) for _ in range(self.n_crop)]
        views_mask = [self.random_mask(x).unsqueeze(dim=0) for _ in range(self.n_mask)]
        views = views_crop + views_mask
        views = torch.cat(views, dim=0)
        return views


def set_few_shot_loader(args):
    root = args.root_dir
    data_transform = RandomCrop(args.n_crop)
    # data_transform = RandomCropAndMask(args.n_crop, args.n_crop)
    shuffle = True
    kwargs = {'num_workers': 0, 'pin_memory': True}

    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'train')
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
    dataset = datasets.ImageFolder(path)

    indices = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    print('get dataset index')
    for i, target in enumerate(tqdm(dataset.targets)):
        classwise_idx[target].append(i)
    print('sample few shot dataset')
    from random import sample
    for i in tqdm(range(args.n_cls)):
        sl = sample(classwise_idx[i], args.n_shot)
        indices.extend(sl)

    dataset = datasets.ImageFolder(path, transform=data_transform)
    dataset = torch.utils.data.Subset(dataset, indices)
    few_shot_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, **kwargs)

    # from torch.utils.data.distributed import DistributedSampler
    # sampler = DistributedSampler(dataset)
    # few_shot_loader = torch.utils.data.DataLoader(dataset, sampler=sampler,
    #                                               batch_size=args.batch_size,
    #                                               shuffle=False, **kwargs)

    return few_shot_loader


def set_few_shot_loader_normal(args):
    root = args.root_dir
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    # data_transform = RandomCropAndMask(args.n_crop, args.n_crop)
    shuffle = True
    kwargs = {'num_workers': 0, 'pin_memory': True}

    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'train')
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
    dataset = datasets.ImageFolder(path)

    indices = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    print('get dataset index')
    for i, target in enumerate(tqdm(dataset.targets)):
        classwise_idx[target].append(i)
    print('sample few shot dataset')
    from random import sample
    for i in tqdm(range(args.n_cls)):
        sl = sample(classwise_idx[i], args.n_shot)
        indices.extend(sl)

    dataset = datasets.ImageFolder(path, transform=data_transform)
    dataset = torch.utils.data.Subset(dataset, indices)
    few_shot_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, **kwargs)

    # from torch.utils.data.distributed import DistributedSampler
    # sampler = DistributedSampler(dataset)
    # few_shot_loader = torch.utils.data.DataLoader(dataset, sampler=sampler,
    #                                               batch_size=args.batch_size,
    #                                               shuffle=False, **kwargs)

    return few_shot_loader

