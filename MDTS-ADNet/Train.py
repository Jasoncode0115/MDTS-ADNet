import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import math
import os.path as osp
from sklearn.metrics import roc_auc_score
import random
import argparse
import time
from tqdm import tqdm
from model.MDTS_ADNet import *
from utils import *
from model.utils import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MDTS-ADNet")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default='Set appropriately based on computer video memory', help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--loss_compact', type=float, default=0.01, help='weight of the feature compactness loss')
    parser.add_argument('--loss_separate', type=float, default=0.01, help='weight of the feature separateness loss')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--method', type=str, default='recon', help='The target task for anomaly detection')
    parser.add_argument('--t_length', type=int, default=1, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='4M-TAD', help='type of dataset')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = args.gpus if args.gpus else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.backends.cudnn.enabled = True

    train_folder_rgb = osp.join(args.dataset_path, args.dataset_type, "training/frames")
    train_folder_of = osp.join(args.dataset_path, args.dataset_type, "training/flows")
    test_folder_rgb = osp.join(args.dataset_path, args.dataset_type, "testing/frames")
    test_folder_of = osp.join(args.dataset_path, args.dataset_type, "testing/flows")

    # 加载数据集
    train_dataset = DataLoader(train_folder_rgb, train_folder_of, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)

    test_dataset = DataLoader(test_folder_rgb, test_folder_of, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)

    train_size = len(train_dataset)
    test_size = len(test_dataset)

    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)


    # Model setting
    model = TwoStreamModel(rgb_channels=3, of_channels=3, t_length=2, latent_dim=128)
    params_encoder = list(model.rgb_model.encoder.parameters()) + list(model.of_model.encoder.parameters())
    params_decoder = list(model.rgb_model.decoder.parameters()) + list(model.of_model.decoder.parameters())
    params = params_encoder + params_decoder + [model.weight_rgb]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.cuda()

    # Report the training process
    log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, 'log.txt')

    # 清空 log.txt 文件内容
    with open(log_file, 'w') as f:
        f.write("")

        # Training
        total_steps = len(train_batch) * args.epochs
        with tqdm(total=total_steps, desc="训练进度", unit="batch", mininterval=1.0) as pbar:
            for epoch in range(args.epochs):
                model.train()
                train_losses = []

                for j, (imgs_rgb, imgs_of) in enumerate(train_batch):
                    imgs_rgb = Variable(imgs_rgb).cuda()
                    imgs_of = Variable(imgs_of).cuda()

                    # Reset hidden states for ConvLSTM
                    model.rgb_model.encoder.convLSTM1.init_hidden(batch_size=imgs_rgb.size(0),
                                                                  image_size=(
                                                                  imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
                    model.rgb_model.encoder.convLSTM2.init_hidden(batch_size=imgs_rgb.size(0),
                                                                  image_size=(
                                                                  imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
                    model.rgb_model.decoder.convLSTM3.init_hidden(batch_size=imgs_rgb.size(0),
                                                                  image_size=(
                                                                  imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
                    model.rgb_model.decoder.convLSTM4.init_hidden(batch_size=imgs_rgb.size(0),
                                                                  image_size=(
                                                                  imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
                    model.of_model.encoder.convLSTM1.init_hidden(batch_size=imgs_rgb.size(0),
                                                                 image_size=(
                                                                 imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
                    model.of_model.encoder.convLSTM2.init_hidden(batch_size=imgs_rgb.size(0),
                                                                 image_size=(
                                                                 imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
                    model.of_model.decoder.convLSTM3.init_hidden(batch_size=imgs_rgb.size(0),
                                                                 image_size=(
                                                                 imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
                    model.of_model.decoder.convLSTM4.init_hidden(batch_size=imgs_rgb.size(0),
                                                                 image_size=(
                                                                 imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))

                    combined_error, rgb_output, of_output, rgb_mu, rgb_logvar, of_mu, of_logvar, weights, weight_rgb = model.forward(
                        imgs_rgb, imgs_of)

                    optimizer.zero_grad()

                    # 使用可学习权重
                    loss = loss_function(rgb_output, of_output, imgs_rgb, imgs_of, rgb_mu, rgb_logvar, of_mu, of_logvar,
                                         weight_rgb)

                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())

                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({'epoch': f'{epoch + 1}/{args.epochs}', '当前损失': f'{loss.item():.6f}'})
                    sys.stdout.flush()
                # 计算并记录这个epoch的平均损失
                avg_loss = np.mean(train_losses)
                with open(log_file, 'a') as f:
                    f.write(f'Epoch {epoch + 1}/{args.epochs} 平均损失: {avg_loss:.6f}\n')

                model.eval()  # Switch model to evaluation mode
                test_loss_epoch = 0.0
                with torch.no_grad():  # Disable gradient calculation in validation/test mode
                    for j, (imgs_rgb, imgs_of) in enumerate(test_batch):
                        imgs_rgb = Variable(imgs_rgb).cuda()
                        imgs_of = Variable(imgs_of).cuda()

                        combined_error, rgb_output, of_output, rgb_mu, rgb_logvar, of_mu, of_logvar, weights, weight_rgb = model.forward(
                            imgs_rgb, imgs_of)

                        loss = loss_function(rgb_output, of_output, imgs_rgb, imgs_of, rgb_mu, rgb_logvar, of_mu,
                                             of_logvar, weight_rgb)

                        test_loss_epoch += loss.item()

                scheduler.step()

                print('----------------------------------------')
                print('Epoch:', epoch + 1)
                print('Train Loss: Reconstruction {:.6f}'.format(avg_loss))
                print('----------------------------------------')

        print('Training is finished')

        # Save the model
        torch.save(model.state_dict(), osp.join(log_dir, 'model.pth'))

