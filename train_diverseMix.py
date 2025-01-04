import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import models.densenet as dn
from utils import ImageNet

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')

parser.add_argument('--in-dataset', default="CIFAR-100", choices=[ 'CIFAR-10','CIFAR-100'],type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')
parser.add_argument('--auxiliary-dataset', default='imagenet',
                    choices=[ 'imagenet'], type=str, help='which auxiliary dataset to use')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--save-epoch', default=10, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,      ###
                    help='mini-batch size (default: 64)')
parser.add_argument('--ood-batch-size', default=128, type=int,###
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--depth', default=40, type=int,
                    help='depth of resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='diverseMix_energy', type=str,
                    help='name of experiment')
parser.add_argument('--workers', default=1, type=int,
                    help='num_workers')
parser.add_argument('--max_class', default=1000, type=int,
                    help='diversity of auxiliary outliers datasets: max class (default: 1000)')
parser.add_argument('--rate', default=1, type=float,
                    help='rate of outliers used (default: 1)')
parser.add_argument('--dataset_path', default='./dataset', type=str,
                    help='path of datasets')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed')
parser.add_argument('--m_in', default=-25, type=float,
                    help='mergin_in')
parser.add_argument('--m_out', default=-7, type=float,
                    help='mergin_out')
parser.add_argument('--alpha', default=4, type=float,
                    help='alpha')
parser.add_argument('--T', default=10, type=float,
                    help='temperature')
parser.add_argument('--beta', default=0.01, type=float, help='beta for out_loss')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)
directory = "checkpoints/{in_dataset}/{name}/{seed}/".format(in_dataset=args.in_dataset, name=args.name,seed=args.seed)
if not os.path.exists(directory):
    os.makedirs(directory)
save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()

assert torch.cuda.is_available()
torch.cuda.set_device(args.gpu)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

def main():

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

    kwargs = {'num_workers': args.workers, 'pin_memory': True}

    if args.in_dataset == "CIFAR-10":
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(args.dataset_path,'cifar10'), train=True, download=False,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(args.dataset_path,'cifar10'), train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        lr_schedule=[50, 75, 90]
        num_classes = 10

    elif args.in_dataset == "CIFAR-100":
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(os.path.join(args.dataset_path,'cifar100'), train=True, download=False,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(os.path.join(args.dataset_path,'cifar100'), train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        lr_schedule=[50, 75, 90]
        num_classes = 100

    if args.auxiliary_dataset == 'imagenet':
        ood_loader = torch.utils.data.DataLoader(
        ImageNet(path=os.path.join(args.dataset_path,'ImagenetRC'),transform=transforms.Compose(
            [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor()]),max_class=args.max_class,use_rate=args.rate),
            batch_size=args.ood_batch_size, shuffle=False, **kwargs)

    if args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    model = model.cuda()
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    ood_criterion = EnergyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr_schedule)

        train_energy(train_loader, ood_loader, model, criterion, ood_criterion, optimizer, epoch)

        prec1 = validate(val_loader, model, criterion, epoch)

        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, epoch + 1)

def train_energy(train_loader_in, train_loader_out, model, criterion, ood_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    nat_in_losses = AverageMeter()
    nat_out_losses = AverageMeter()
    nat_top1 = AverageMeter()

    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))

    model.train()

    end = time.time()
    for i, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):

        in_len = len(in_set[0])
        out_len = len(out_set[0])
        target = in_set[1]
        target = target.to(torch.int64).cuda()
        in_input=in_set[0].cuda()
        out_input = out_set[0].cuda()

        # use diverseMix
        model.eval()
        with torch.no_grad():
            energy=torch.logsumexp(model(out_input),dim=1)
            shuffle_idx = torch.randperm(out_input.size(0)).cuda()
            confidence=torch.softmax(torch.stack([energy,energy[shuffle_idx]],dim=1)/args.T,dim=1)
            beta_dist=torch.distributions.Beta(args.alpha*confidence[:,0]+0.01, args.alpha*confidence[:,1]+0.01)
            weight = beta_dist.rsample().view(-1, 1, 1, 1)
            out_input = weight * out_input + (1 - weight) * out_input[shuffle_idx]
        model.train()

        input = torch.cat((in_input, out_input), 0)

        output = model(input)

        nat_in_output = output[:in_len]
        nat_in_loss = criterion(nat_in_output, target)

        nat_out_output = output[in_len:]
        nat_out_loss = ood_criterion(nat_out_output, nat_in_output)

        nat_prec1 = accuracy(nat_in_output.data, target, topk=(1,))[0]
        nat_in_losses.update(nat_in_loss.data, in_len)
        nat_out_losses.update(nat_out_loss.data, out_len)
        nat_top1.update(nat_prec1, in_len)

        loss = nat_in_loss + args.beta * nat_out_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'In Loss {in_loss.val:.4f} ({in_loss.avg:.4f})\t'
                  'Out Loss {out_loss.val:.4f} ({out_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader_in), batch_time=batch_time,
                      in_loss=nat_in_losses, out_loss=nat_out_losses, top1=nat_top1))

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.to(torch.int64).cuda()

        output = model(input)
        loss = criterion(output, target)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))


def save_checkpoint(state, epoch):
    """Saves checkpoint to disk"""
    directory = "checkpoints/{in_dataset}/{name}/{seed}/".format(in_dataset=args.in_dataset, name=args.name, seed=args.seed)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'checkpoint_{}.pth.tar'.format(epoch)
    torch.save(state, filename)

class EnergyLoss(nn.Module):
    def __init__(self):
        super(EnergyLoss, self).__init__()
    def forward(self, x_out,x_in):
        E_out = -torch.logsumexp(x_out,dim=1)
        E_in  = -torch.logsumexp(x_in,dim=1)
        return torch.pow(F.relu(E_in - args.m_in), 2).mean() + torch.pow(F.relu(args.m_out - E_out), 2).mean()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr_schedule=[50, 75, 90]):
    """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""
    lr = args.lr
    if epoch >= lr_schedule[0]:
        lr *= 0.1
    if epoch >= lr_schedule[1]:
        lr *= 0.1
    if epoch >= lr_schedule[2]:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
