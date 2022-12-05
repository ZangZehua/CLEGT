import os

import torch
import argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.models
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from utils import save_config_file, accuracy, save_checkpoint
from torch.cuda.amp import GradScaler, autocast
import logging





parser = argparse.ArgumentParser(description='PyTorch Supervisor')
parser.add_argument('--dataset-name', default='cifar100', help='dataset name', choices=['stl10', 'cifar10', 'cifar100'])
parser.add_argument('--data', default='./datasets', help='path to dataset')
parser.add_argument('--epochs', default=100, type=int, help='sl learning epochs')
parser.add_argument('--batch-size', default=30, type=int, help='learning batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--arch', default='resnet18', help='model architecture')


def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.STL10('./datasets', split='train', download=download,
                                   transform=transforms.Compose([
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.441, 0.427, 0.385],
                                                            std=[0.231, 0.226, 0.224])
                                   ]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.STL10('./datasets', split='test', download=download,
                                  transform=transforms.Compose([
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.441, 0.427, 0.385],
                                                           std=[0.231, 0.226, 0.224])
                                  ]))

    test_loader = DataLoader(test_dataset, batch_size=2 * batch_size,
                             num_workers=10, drop_last=False, shuffle=shuffle)

    return train_loader, test_loader


def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR10('./datasets', train=True, download=download,
                                     transform=transforms.Compose([
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(0.5),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.491, 0.482, 0.446], [0.202, 0.199, 0.201])
                                     ]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=16, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.CIFAR10('./datasets', train=False, download=download,
                                    transform=transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.491, 0.482, 0.446], [0.202, 0.199, 0.201])
                                    ]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=16, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def get_cifar100_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR100('./datasets', train=True, download=download,
                                      transform=transforms.Compose([
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.507, 0.487, 0.441], [0.201, 0.198, 0.202])
                                      ]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=16, pin_memory=True, drop_last=True, shuffle=shuffle)

    test_dataset = datasets.CIFAR100('./datasets', train=False, download=download,
                                     transform=transforms.Compose([
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(0.5),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.507, 0.487, 0.441], [0.201, 0.198, 0.202])
                                     ]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=16, pin_memory=True, drop_last=True, shuffle=shuffle)
    return train_loader, test_loader


def main():
    args = parser.parse_args()
    comment = '_sl_'+args.dataset_name+'_'+args.arch+'_'+str(args.epochs)+'epochs'
    writer = SummaryWriter(comment=comment)
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'sl_training.log'), level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args.dataset_name)
    if args.dataset_name == 'stl10':
        train_loader, test_loader = get_stl10_data_loaders(download=False, shuffle=True, batch_size=args.batch_size)
        num_classes = 10
    elif args.dataset_name == 'cifar10':
        train_loader, test_loader = get_cifar10_data_loaders(download=False, shuffle=True, batch_size=args.batch_size)
        num_classes = 10
    elif args.dataset_name == 'cifar100':
        train_loader, test_loader = get_cifar100_data_loaders(download=False, shuffle=True, batch_size=args.batch_size)
        num_classes = 100

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = GradScaler(enabled=args.fp16_precision)
    epochs = args.epochs
    logging.info(f"Start SimCLR training for {epochs} epochs.")
    n_iter = 0
    for epoch in range(epochs):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(tqdm(train_loader)):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            with autocast(enabled=args.fp16_precision):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if n_iter % 100 == 0:
                top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
                writer.add_scalar('loss', loss, global_step=n_iter)
                writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                writer.add_scalar('acc/top5', top5[0], global_step=n_iter)

            n_iter += 1

        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

        logging.debug(
            f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

        print(
            f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

    logging.info("Training has finished.")
    # save model checkpoints
    checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(args.epochs)
    save_checkpoint({
        'epoch': args.epochs,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(writer.log_dir, checkpoint_name))
    logging.info(f"Model checkpoint and metadata has been saved at {writer.log_dir+'_sl'}.")


if __name__ == "__main__":
    main()
