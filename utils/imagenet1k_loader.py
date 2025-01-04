import numpy as np
import torch
import os
import pickle
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def Imagenet200_800_loader(ID_class=True,shuffle=True,train=True, max_count=500,preprocess=None,batch_size=256,num_workers=4,root='./datasets'):
    if ID_class:
        path = os.path.join('./utils/label_split','ID_class.txt')
    else:
        path = os.path.join('./utils/label_split', 'OOD_class.txt')
    with open(path, 'r') as label_file:
        selected_classes = label_file.read().splitlines()

    class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

    if train:
        path = os.path.join(root, 'ImageNet1K', 'train')
    else:
        path = os.path.join(root, 'ImageNet1K', 'val')
    dataset = datasets.ImageFolder(path, transform=preprocess)
    filtered_samples = []
    class_counts = {cls: 0 for cls in selected_classes}
    class2idx={ selected_classes[i] : i for i in range(len(selected_classes))}
    for path, label in dataset.samples:
        cls = dataset.classes[label]
        if cls in selected_classes and class_counts[cls] < max_count:
            filtered_samples.append((path, class2idx[cls]))
            class_counts[cls] += 1
    classes=list(class_counts.keys())

    dataset.classes=classes
    dataset.samples = filtered_samples
    dataset.targets = [label for path, label in filtered_samples]

    dataset.class_to_idx = class_to_idx
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    return data_loader
