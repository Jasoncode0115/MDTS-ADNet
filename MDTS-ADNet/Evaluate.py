
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
import random
import glob
import argparse
from tqdm import tqdm

from utils import *
from model.utils import DataLoader
from model.MDTS_ADNet import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MDTS-ADNet")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default='Set appropriately based on computer video memory', help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--method', type=str, default='recon', help='The target task for anomaly detection')
    parser.add_argument('--t_length', type=int, default=1, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--alpha', type=float, default='Set up appropriately based on your upcoming task', help='weight for the anomality score')
    parser.add_argument('--th', type=float, default='Set up appropriately based on your upcoming task', help='threshold for test updating')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='4M-TAD', help='type of dataset')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
    parser.add_argument('--model_dir', type=str, default='./exp/4M-TAD/recon/log/model.pth', help='directory of model')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        gpus = ""
        for i in range(len(args.gpus)):
            gpus = gpus + args.gpus[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

    torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

    test_folder_rgb = args.dataset_path + "/" + args.dataset_type + "/testing/frames"
    test_folder_of = args.dataset_path + "/" + args.dataset_type + "/testing/flows"

    # Loading dataset
    test_dataset = DataLoader(test_folder_rgb, test_folder_of, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)

    test_size = len(test_dataset)

    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    model = TwoStreamModel(rgb_channels=3, of_channels=3, t_length=2, latent_dim=128)  # 替换为实际参数

    model.load_state_dict(torch.load(args.model_dir))
    model.cuda()

    labels_rgb = np.load('./data/frame_labels_' + args.dataset_type + '.npy')
    labels_of = np.load('./data/frame_labels_' + args.dataset_type + '.npy')

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder_rgb, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list_rgb = []
    labels_list_of = []
    label_length = 0
    psnr_list_rgb = {}
    psnr_list_of = {}

    print('Evaluation of', args.dataset_type)

    pbar = tqdm(total=len(test_batch), desc="Testing Progress")

    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        labels_list_rgb = np.append(labels_list_rgb,
                                    labels_rgb[0][label_length:videos[video_name]['length'] + label_length])
        labels_list_of = np.append(labels_list_of, labels_of[0][label_length:videos[video_name]['length'] + label_length])
        label_length += videos[video_name]['length']
        psnr_list_rgb[video_name] = []
        psnr_list_of[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    model.eval()

    for k, (imgs_rgb, imgs_of) in enumerate(test_batch):
        if k == label_length:
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
            model.rgb_model.encoder.convLSTM1.init_hidden(batch_size=imgs_rgb.size(0),
                                                          image_size=(imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
            model.rgb_model.encoder.convLSTM2.init_hidden(batch_size=imgs_rgb.size(0),
                                                          image_size=(imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
            model.rgb_model.decoder.convLSTM3.init_hidden(batch_size=imgs_rgb.size(0),
                                                          image_size=(imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
            model.rgb_model.decoder.convLSTM4.init_hidden(batch_size=imgs_rgb.size(0),
                                                          image_size=(imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
            model.of_model.encoder.convLSTM1.init_hidden(batch_size=imgs_rgb.size(0),
                                                         image_size=(imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
            model.of_model.encoder.convLSTM2.init_hidden(batch_size=imgs_rgb.size(0),
                                                         image_size=(imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
            model.of_model.decoder.convLSTM3.init_hidden(batch_size=imgs_rgb.size(0),
                                                         image_size=(imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
            model.of_model.decoder.convLSTM4.init_hidden(batch_size=imgs_rgb.size(0),
                                                         image_size=(imgs_rgb.size(2) // 16, imgs_rgb.size(3) // 16))
        imgs_rgb = Variable(imgs_rgb).cuda()
        imgs_of = Variable(imgs_of).cuda()

        combined_error, rgb_output, of_output, rgb_mu, rgb_logvar, of_mu, of_logvar, weights, weight_rgb = model.forward(
            imgs_rgb, imgs_of)

        mse_rgb = torch.mean(F.mse_loss((rgb_output[0] + 1) / 2, (imgs_rgb[0] + 1) / 2)).item()
        mse_of = torch.mean(F.mse_loss((of_output[0] + 1) / 2, (imgs_of[0] + 1) / 2)).item()

        psnr_list_rgb[videos_list[video_num].split('/')[-1]].append(psnr(mse_rgb))
        psnr_list_of[videos_list[video_num].split('/')[-1]].append(psnr(mse_of))

        pbar.update(1)

    pbar.close()

    anomaly_score_total_list_rgb = []
    anomaly_score_total_list_of = []
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        anomaly_score_total_list_rgb += anomaly_score_list(psnr_list_rgb[video_name])
        anomaly_score_total_list_of += anomaly_score_list(psnr_list_of[video_name])

    anomaly_score_total_list_rgb = np.asarray(anomaly_score_total_list_rgb)
    anomaly_score_total_list_of = np.asarray(anomaly_score_total_list_of)

    accuracy_rgb = AUC(anomaly_score_total_list_rgb, np.expand_dims(1 - labels_list_rgb, 0))
    accuracy_of = AUC(anomaly_score_total_list_of, np.expand_dims(1 - labels_list_of, 0))

    combined_anomaly_scores = (anomaly_score_total_list_rgb + anomaly_score_total_list_of) / 2
    accuracy_combined = AUC(combined_anomaly_scores, np.expand_dims(1 - labels_list_rgb, 0))

    print('The result of ', args.dataset_type)
    print('AUC for RGB: ', accuracy_rgb * 100, '%')
    print('AUC for Optical Flow: ', accuracy_of * 100, '%')
    print('Combined AUC: ', accuracy_combined * 100, '%')

