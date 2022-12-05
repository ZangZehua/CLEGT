import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import BackBoneNet
# from simclr import SimCLR
from simclr_bt import SimClrBT

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10', 'cifar100'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--gpu-index', default=3, type=int, help='Gpu index.')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--alpha', default=1, type=float, help='coefficient of the simclr loss')
parser.add_argument('--beta', default=0, type=float, help='coefficient of the barlowtwins loss')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')


def main():
    args = parser.parse_args()
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        # 数据输入维度或类型变化不大设置cudnn.benchmark为true可以增加运行效率
        # 若输入数据在每个iteration都变化cudnn会每次寻找一次最优配置，反而会降低效率，所以cudnn的确定性设置为true
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = BackBoneNet(base_model=args.arch, out_dim=args.out_dim, projector=args.projector)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        model = model.to(args.device)
        simclr_bt = SimClrBT(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr_bt.train(train_loader)


if __name__ == "__main__":
    main()
