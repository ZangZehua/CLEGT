import logging
import os
import sys

import torch
import torch.nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SimClrBT(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        comment = '_ssl_' + self.args.dataset_name + '_' + self.args.arch + '_epochs_' + str(self.args.epochs) + '_alpha_' + str(
            self.args.alpha) + '_beta_' + str(self.args.beta)
        self.writer = SummaryWriter(comment=comment)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        # batchsize为N，获得一个[]维张量为[0-N-1,0-N-1]
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0)
        # 张量在0维和1维扩充得一个2Nx2N的单位阵
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        # 从标签和相似度矩阵中删除对角线
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def barlowtwins_loss(self, features1, features2, feature_dim):
        # bn = torch.nn.BatchNorm1d(feature_dim)
        bn = self.model.bn
        c = bn(features1).T @ bn(features2)
        c.div_(self.args.batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for (y1, y2, y3, y4), _ in tqdm(train_loader):
                y1 = y1.to(self.args.device)
                # print(self.args.device)
                y2 = y2.to(self.args.device)
                simclr_images = torch.cat([y3, y4], dim=0)
                simclr_images = simclr_images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    # simclr loss
                    features = self.model(simclr_images)
                    # # print(features.shape)
                    logits, labels = self.info_nce_loss(features)
                    simclr_loss = self.criterion(logits, labels)
                    # barlowtwins loss
                    feature1 = self.model(y1)
                    feature2 = self.model(y2)
                    barlowtwins_loss = self.barlowtwins_loss(feature1, feature2, self.model.sizes[-1])

                self.optimizer.zero_grad()
                # alpha为两个loss的联合系数
                loss = self.args.alpha * simclr_loss + self.args.beta * barlowtwins_loss
                # loss = self.args.alpha * barlowtwins_loss
                # loss = simclr_loss
                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            print(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        print("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        print(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
