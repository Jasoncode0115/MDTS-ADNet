import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):
    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def normalize_img(img):
    img_re = copy.copy(img)
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    return img_re

class WeightModel(nn.Module):
    def __init__(self):
        super(WeightModel, self).__init__()
        # Adaptive fusion weights as a learnable parameter
        self.weight_rgb = nn.Parameter(torch.tensor(0.5))  # Initial weight

    def forward(self):
        return F.softmax(self.weight_rgb, dim=0)

def point_score(outputs_rgb, outputs_of, imgs_rgb, imgs_of, mu_rgb, logvar_rgb, mu_of, logvar_of, weight_model):
    weight_rgb = weight_model()  # Get the learned weight

    # Calculate reconstruction losses
    BCE_rgb = F.mse_loss(outputs_rgb, imgs_rgb, reduction='mean')
    BCE_of = F.mse_loss(outputs_of, imgs_of, reduction='mean')

    # Calculate KL divergences
    KLD_rgb = -0.5 * torch.sum(1 + logvar_rgb - mu_rgb.pow(2) - logvar_rgb.exp())
    KLD_of = -0.5 * torch.sum(1 + logvar_of - mu_of.pow(2) - logvar_of.exp())

    # Compute normalizers for RGB and OF streams
    normal_rgb = (1 - torch.exp(-BCE_rgb))
    normal_of = (1 - torch.exp(-BCE_of))

    # Combine scores from both streams
    score_rgb = (torch.sum(normal_rgb * BCE_rgb) / torch.sum(normal_rgb)).item()
    score_of = (torch.sum(normal_of * BCE_of) / torch.sum(normal_of)).item()

    # Weighted average score
    score = weight_rgb * score_rgb + (1 - weight_rgb) * score_of
    return score

def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr - min_psnr)))

def anomaly_score_list(psnr_list):
    return [anomaly_score(psnr, np.max(psnr_list), np.min(psnr_list)) for psnr in psnr_list]

def anomaly_score_list_inv(psnr_list):
    return [anomaly_score_inv(psnr, np.max(psnr_list), np.min(psnr_list)) for psnr in psnr_list]

def AUC(anomal_scores, labels):
    try:
        frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
        return frame_auc
    except ValueError as e:
        logging.error(f"Error calculating AUC: {e}")
        return None

def score_sum(list1, list2, alpha):
    return [alpha * l1 + (1 - alpha) * l2 for l1, l2 in zip(list1, list2)]

def loss_function(outputs_rgb, outputs_of, imgs_rgb, imgs_of, mu_rgb, logvar_rgb, mu_of, logvar_of, weight_model):
    weight_rgb = weight_model  # 直接使用 weight_model 的值

    # Calculate reconstruction losses
    BCE_rgb = F.mse_loss(outputs_rgb, imgs_rgb, reduction='mean')
    BCE_of = F.mse_loss(outputs_of, imgs_of, reduction='mean')

    # Calculate KL divergences
    KLD_rgb = -0.5 * torch.sum(1 + logvar_rgb - mu_rgb.pow(2) - logvar_rgb.exp())
    KLD_of = -0.5 * torch.sum(1 + logvar_of - mu_of.pow(2) - logvar_of.exp())

    # Calculate total loss
    total_loss = weight_rgb * (BCE_rgb + KLD_rgb) + (1 - weight_rgb) * (BCE_of + KLD_of)

    return total_loss


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
