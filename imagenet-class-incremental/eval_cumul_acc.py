#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import sys
import copy
import argparse
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle
from scipy.spatial.distance import cdist

import utils_pytorch
from utils_imagenet.utils_dataset import split_images_labels
from utils_imagenet.utils_dataset import merge_images_labels
from utils_incremental.compute_features import compute_features
from utils_incremental.compute_accuracy import compute_accuracy
from utils_incremental.compute_confusion_matrix import compute_confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default='data/seed_1993_subset_100_imagenet/data', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--nb_cl', default=10, type=int, \
    help='Classes per group')
parser.add_argument('--ckp_prefix', \
    default='checkpoint/class_incremental_imagenet_nb_cl_fg_50_nb_cl_10_nb_protos_200_run_0_', \
    type=str)
parser.add_argument('--order', \
    default='./checkpoint/seed_1993_subset_100_imagenet_order_run_0.pkl', \
    type=str)
parser.add_argument('--nb_cl_fg', default=50, type=int, \
    help='the number of classes in first group')
args = parser.parse_args()
print(args)

order = utils_pytorch.unpickle(args.order)
order_list = list(order)
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),
# ])
# evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
#                                        download=False, transform=transform_test)
# input_data = evalset.test_data
# input_labels = evalset.test_labels
# map_input_labels = np.array([order_list.index(i) for i in input_labels])
valdir = os.path.join(args.datadir, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
evalset =  datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
input_data, input_labels = split_images_labels(evalset.imgs)
map_input_labels = np.array([order_list.index(i) for i in input_labels])
# evalset.test_labels = map_input_labels
# evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
#     shuffle=False, num_workers=2)

cnn_cumul_acc = []
icarl_cumul_acc = []
ncm_cumul_acc = []
num_classes = []
nb_cl = args.nb_cl
start_iter = int(args.nb_cl_fg/nb_cl)-1
for iteration in range(start_iter, int(args.num_classes/nb_cl)):
    # print("###########################################################")
    # print("For iteration {}".format(iteration))
    # print("###########################################################")
    ckp_name = '{}iteration_{}_model.pth'.format(args.ckp_prefix, iteration)
    class_means_name = '{}iteration_{}_class_means.pth'.format(args.ckp_prefix, iteration)
    if not os.path.exists(ckp_name):
        break
    tg_model = torch.load(ckp_name)
    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
    class_means = torch.load(class_means_name)
    current_means = class_means[:, order[:(iteration+1)*nb_cl]]
    indices = np.array([i in range(0, (iteration+1)*nb_cl) for i in map_input_labels])
    # evalset.test_data = input_data[indices]
    # evalset.test_labels = map_input_labels[indices]
    # print('Max and Min of valid labels: {}, {}'.format(min(evalset.test_labels), max(evalset.test_labels)))
    current_eval_set = merge_images_labels(input_data[indices], map_input_labels[indices])
    evalset.imgs = evalset.samples = current_eval_set
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
        shuffle=False, num_workers=8, pin_memory=True)
    print("###########################################################")
    acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, print_info=True)
    print("###########################################################")
    cnn_cumul_acc.append(acc[0])
    icarl_cumul_acc.append(acc[1])
    ncm_cumul_acc.append(acc[2])
    num_classes.append((iteration+1)*nb_cl)

print("###########################################################")
print(' CNN  acc: \t iCaRL acc \t NCM acc')
print("###########################################################")
for i in range(len(cnn_cumul_acc)):
    print("{:.2f} ".format(cnn_cumul_acc[i]), end='')
print("[{:.2f}] ".format(np.mean(cnn_cumul_acc[-1])), end='')
print("[{:.2f}] ".format(np.mean(cnn_cumul_acc)), end='')
print("[{:.2f}] ".format(np.sum(np.array(cnn_cumul_acc)*np.array(num_classes)) / np.sum(num_classes)), end='')
print("")
for i in range(len(icarl_cumul_acc)):
    print("{:.2f} ".format(icarl_cumul_acc[i]), end='')
print("[{:.2f}] ".format(np.mean(icarl_cumul_acc[-1])), end='')
print("[{:.2f}] ".format(np.mean(icarl_cumul_acc)), end='')
print("[{:.2f}] ".format(np.sum(np.array(icarl_cumul_acc)*np.array(num_classes)) / np.sum(num_classes)), end='')
print("")
for i in range(len(cnn_cumul_acc)):
    print("{:.2f} ".format(ncm_cumul_acc[i]), end='')
print("[{:.2f}] ".format(np.mean(ncm_cumul_acc[-1])), end='')
print("[{:.2f}] ".format(np.mean(ncm_cumul_acc)), end='')
print("[{:.2f}] ".format(np.sum(np.array(ncm_cumul_acc)*np.array(num_classes)) / np.sum(num_classes)), end='')
print("")
print("###########################################################")
print("")
print('Number of classes', num_classes)
print("###########################################################")
print("Final acc on all classes")
print("CNN:{:.2f}\t iCaRL:{:.2f}\t NCM:{:.2f}".format(cnn_cumul_acc[-1], icarl_cumul_acc[-1], ncm_cumul_acc[-1]))
print("###########################################################")
print("Average acc in each phase")
print("CNN:{:.2f}\t iCaRL:{:.2f}\t NCM:{:.2f}".format(np.mean(cnn_cumul_acc), np.mean(icarl_cumul_acc), np.mean(ncm_cumul_acc)))
print("###########################################################")
print("Weighted average acc in each phase")
print("CNN:{:.2f}\t iCaRL:{:.2f}\t NCM:{:.2f}".format(
    np.sum(np.array(cnn_cumul_acc)*np.array(num_classes)) / np.sum(num_classes),
    np.sum(np.array(icarl_cumul_acc)*np.array(num_classes)) / np.sum(num_classes),
    np.sum(np.array(ncm_cumul_acc)*np.array(num_classes)) / np.sum(num_classes)
    ))
print("###########################################################")