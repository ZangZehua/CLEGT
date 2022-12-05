import os

import torch
import torch.nn
import argparse
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import logging

from tqdm import tqdm

from utils import save_config_file, accuracy, save_checkpoint

parser = argparse.ArgumentParser("Linear Probing")
parser.add_argument("--dataset-name", default='stl10', choices=['stl10', 'cifar100'], help="linear probe dataset")
parser.add_argument("--batch-size", type=int, default=128, help='mini batch size')
parser.add_argument("--epochs", type=int, default=100, help='training epochs')
parser.add_argument("--lr", type=float, default=3e-4, help='learning rate')
parser.add_argument("--weight-decay", type=float, default='0.0008', help='weight decay')
parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
parser.add_argument("--arch", default='resnet18', help='backbone model structure')
parser.add_argument("--model-path",
                    default='/home/jqr/BTandCL/runs/Dec02_03-24-21_master_ssl_stl10_resnet18_epochs_200_alpha_1_beta_0/checkpoint_0200.pth.tar',
                    help='path to load model')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')


def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.STL10('./datasets', split='train', download=download,
                                   transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.STL10('./datasets', split='test', download=download,
                                  transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2 * batch_size,
                             num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def get_cifar100_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR100('./datasets', train=True, download=download,
                                      transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.CIFAR10('./datasets', train=False, download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2 * batch_size,
                             num_workers=100, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def main():
    args = parser.parse_args()
    comment = '_linearprobing_' + args.dataset_name + '_' + args.arch + '_' + str(
        args.epochs) + 'epochs_' + args.model_path
    writer = SummaryWriter(comment=comment)
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'linearprobing.log'), level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args.dataset_name)
    if args.dataset_name == 'stl10':
        train_loader, test_loader = get_stl10_data_loaders(download=False, shuffle=True, batch_size=args.batch_size)
        num_classes = 10
    elif args.dataset_name == 'cifar100':
        train_loader, test_loader = get_cifar100_data_loaders(download=False, shuffle=True, batch_size=args.batch_size)
        num_classes = 100

    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone'):
                state_dict[k[len('backbone.'):]] = state_dict[k]
        del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    print(log)
    assert log.missing_keys == ['fc.weight', 'fc.bias']

    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.9)

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
    logging.info(f"Model checkpoint and metadata has been saved at {writer.log_dir + '_linearprobing'}.")


if __name__ == "__main__":
    main()
