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
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_pytorch import *

cur_features = []
ref_features = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def incremental_train_and_eval_LF(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iteration, \
            lamda, \
            fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #trainset.train_data = X_train.astype('uint8')
    #trainset.train_labels = Y_train
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
    #    shuffle=True, num_workers=2)
    #testset.test_data = X_valid.astype('uint8')
    #testset.test_labels = Y_valid
    #testloader = torch.utils.data.DataLoader(testset, batch_size=100,
    #    shuffle=False, num_workers=2)
    #print('Max and Min of train labels: {}, {}'.format(min(Y_train), max(Y_train)))
    #print('Max and Min of valid labels: {}, {}'.format(min(Y_valid), max(Y_valid)))

    if iteration > start_iteration:
        ref_model.eval()
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
    for epoch in range(epochs):
        #train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    #m.weight.requires_grad = False
                    #m.bias.requires_grad = False
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)
            if iteration == start_iteration:
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                ref_outputs = ref_model(inputs)
                loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
                    torch.ones(inputs.shape[0]).to(device)) * lamda
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss = loss1 + loss2
            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            if iteration > start_iteration:
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #if iteration == 0:
            #    msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
            #    (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            #else:
            #    msg = 'Loss1: %.3f Loss2: %.3f Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
            #    (loss1.item(), loss2.item(), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            #progress_bar(batch_idx, len(trainloader), msg)
        if iteration == start_iteration:
            print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(\
                len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        else:
            print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f},\
                Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader), \
                train_loss1/(batch_idx+1), train_loss2/(batch_idx+1),
                train_loss/(batch_idx+1), 100.*correct/total))

        #eval
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    if iteration > start_iteration:
        print("Removing register_forward_hook")
        handle_ref_features.remove()
        handle_cur_features.remove() 
    return tg_model