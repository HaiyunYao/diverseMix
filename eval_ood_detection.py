from __future__ import print_function
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import models.densenet as dn
import models.wideresnet as wn
import utils.svhn_loader as svhn
import numpy as np
import time
from utils import imagenet1k_loader
from models.resnet18_224x224 import ResNet18_224x224
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--in-dataset', default="CIFAR-100", type=str, help='in-distribution dataset')
parser.add_argument('--name', default="diverseMix/0", type=str, help='the name of the model trained')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')
parser.add_argument('--gpu', default = 0, type = int, help='gpu index')
parser.add_argument('--in-dist-only', help='only evaluate in-distribution', action='store_true')
parser.add_argument('--out-dist-only', help='only evaluate out-distribution', action='store_true')
parser.add_argument('--method', default='energy', type=str, help='scoring function')
parser.add_argument('--cal-metric', help='calculate metric directly', action='store_true')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--depth', default=40, type=int,
                    help='depth of resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')
parser.add_argument('--dataset_path', default='/datasets', type=str,
                    help='path of datasets')
parser.add_argument('--ood_dataset_path', default='/datasets/ood_datasets', type=str,
                    help='path of datasets')
parser.set_defaults(argument=True)

args = parser.parse_args()

assert torch.cuda.is_available()
torch.cuda.set_device(args.gpu)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

def get_msp_score(inputs, model, method_args):
    with torch.no_grad():
        outputs = model(inputs)
    scores = np.max(F.softmax(outputs, dim=1).detach().cpu().numpy(), axis=1)

    return scores

def get_ood_score(inputs, model, method_args):
    with torch.no_grad():
        outputs = model(inputs)

    scores = -1.0 * (F.softmax(outputs, dim=1)[:,-1]).float().detach().cpu().numpy()
    return scores

def get_energy_score(inputs, model, method_args):
    with torch.no_grad():
        outputs = model(inputs)

    return torch.logsumexp(outputs,dim=1).float().detach().cpu().numpy()


def get_score(inputs, model, method, method_args, raw_score=False):
    if method == "msp":
        scores = get_msp_score(inputs, model, method_args)
    elif method=="energy":
        scores = get_energy_score(inputs, model, method_args)
    elif method == "ntom":
        scores = get_ood_score(inputs, model, method_args)
    else:
        print('error')
    return scores

def eval_ood_detector(base_dir, in_dataset, out_datasets, batch_size, method, method_args, name, epochs, mode_args):

    in_save_dir = os.path.join(base_dir, in_dataset, method, name, 'nat')

    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.model_arch == 'resnet18':
        size=224
    else:
        size=32

    if in_dataset == "CIFAR-10":
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        testset = torchvision.datasets.CIFAR10(root=os.path.join(args.dataset_path,'cifar10'), train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
        num_classes = 10
        num_reject_classes = 0
    elif in_dataset == "CIFAR-100":
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        testset = torchvision.datasets.CIFAR100(root=os.path.join(args.dataset_path,'cifar100'), train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
        num_classes = 100
        num_reject_classes = 0

    elif args.in_dataset == "imagenet-200-224x224":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),normalize])
        
        testloaderIn = imagenet1k_loader.Imagenet200_800_loader(ID_class=True, shuffle=False,train=False, max_count=1400,preprocess=transform,batch_size=args.batch_size, num_workers=4, root=args.dataset_path)
        num_classes = 200
        num_reject_classes = 0
    elif args.in_dataset=="imagenet-1k":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(os.path.join(args.ID_dataset_path,"ImageNet1K"),'val'), transform=preprocess)
        testloaderIn = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes=1000
        num_reject_classes = 0

    if method=="ntom":
        num_reject_classes = 1

    method_args['num_classes'] = num_classes

    if args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes + num_reject_classes, normalizer=normalizer)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(args.depth, num_classes + num_reject_classes, widen_factor=args.width, normalizer=normalizer)
    elif args.model_arch == 'resnet18':
            model = ResNet18_224x224(num_classes=num_classes + num_reject_classes)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    checkpoint = torch.load("./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=in_dataset, name=name, epochs=epochs))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.cuda()

    if not mode_args['out_dist_only']:
        t0 = time.time()

        f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')

    # In-distribution
        print("Processing in-distribution images")

        N = len(testloaderIn.dataset)
        count = 0
        for j, data in enumerate(testloaderIn):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images

            scores = get_score(inputs, model, method, method_args)

            for score in scores:
                f1.write("{}\n".format(score))

            if method == "rowl":
                outputs = F.softmax(model(inputs), dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)
            else:
                outputs = F.softmax(model(inputs)[:, :num_classes], dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)

            for k in range(preds.shape[0]):
                g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()
            
            if count>10000:
                break

        f1.close()
        g1.close()

    if mode_args['in_dist_only']:
        return

    for out_dataset in out_datasets:

        out_save_dir = os.path.join(in_save_dir, out_dataset)

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)



        if args.in_dataset=="imagenet-200-224x224":
            if out_dataset == 'dtd':
                testsetout = torchvision.datasets.ImageFolder(root=os.path.join(args.ood_dataset_path,'dtd/images'),
                                            transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),normalize]))
                testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False,
                                                        num_workers=8)
            elif out_dataset == 'places365':
                testsetout = torchvision.datasets.ImageFolder(root=os.path.join(args.ood_dataset_path,'Places365/test_subset'),
                                            transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),normalize]))
                testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False,
                                                        num_workers=8)
            else:
                testsetout = torchvision.datasets.ImageFolder(os.path.join(args.ood_dataset_path,"{}".format(out_dataset)),
                                            transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),normalize]))
                testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size,
                                                shuffle=False, num_workers=8)
        else:
            if out_dataset == 'SVHN':
                testsetout = svhn.SVHN(os.path.join(args.ood_dataset_path,'svhn'), split='test',
                                    transform=transforms.ToTensor(), download=False)
                testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
            elif out_dataset == 'dtd':
                testsetout = torchvision.datasets.ImageFolder(root=os.path.join(args.ood_dataset_path,"dtd/images"),
                                            transform=transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()]))
                testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True,
                                                        num_workers=2)
            elif out_dataset == 'places365':
                testsetout = torchvision.datasets.ImageFolder(root=os.path.join(args.ood_dataset_path,"Places365/test_subset"),
                                            transform=transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()]))
                testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True,
                                                        num_workers=2)
            else:
                testsetout = torchvision.datasets.ImageFolder(os.path.join(args.ood_dataset_path,"{}".format(out_dataset)),
                                            transform=transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()]))
                testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

    # Out-of-Distributions
        t0 = time.time()
        print("Processing out-of-distribution images")

        N = len(testloaderOut.dataset)
        count = 0
        for j, data in enumerate(testloaderOut):

            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]
            inputs = images

            scores = get_score(inputs, model, method, method_args)

            for score in scores:
                f2.write("{}\n".format(score))

            count += curr_batch_size
            if count>10000:
                break
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

        f2.close()

    return

if __name__ == '__main__':
    method_args = dict()
    adv_args = dict()
    mode_args = dict()

    mode_args['in_dist_only'] = args.in_dist_only
    mode_args['out_dist_only'] = args.out_dist_only
    if args.in_dataset in ["CIFAR-10","CIFAR-100"]:
        out_datasets = ['SVHN','LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']
    elif args.in_dataset in ["imagenet-200-224x224"]:
        out_datasets= ['openimage_o_3','ssb_hard_3', 'dtd','iNaturalist','ninco']
    eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs, mode_args)


