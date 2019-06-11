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
old_scores = []
new_scores = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs

def incremental_train_and_eval_AMR_LF(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iteration, \
            lamda, \
            dist, K, lw_mr, \
            weight_per_class=None, device=None):
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
        num_old_classes = ref_model.fc.out_features
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
        handle_old_scores_bs = tg_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
        handle_new_scores_bs = tg_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)
    for epoch in range(epochs):
        #train
        tg_model.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
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
                #################################################
                #scores before scale, [-1, 1]
                outputs_bs = torch.cat((old_scores, new_scores), dim=1)
                assert(outputs_bs.size()==outputs.size())
                #print("outputs_bs:", outputs_bs.size(), outputs_bs)
                #print("targets:", targets.size(), targets)
                #get groud truth scores
                gt_index = torch.zeros(outputs_bs.size()).to(device)
                gt_index = gt_index.scatter(1, targets.view(-1,1), 1).ge(0.5)
                gt_scores = outputs_bs.masked_select(gt_index)
                #print("gt_index:", gt_index.size(), gt_index)
                #print("gt_scores:", gt_scores.size(), gt_scores)
                #get top-K scores on none gt classes
                none_gt_index = torch.zeros(outputs_bs.size()).to(device)
                none_gt_index = none_gt_index.scatter(1, targets.view(-1,1), 1).le(0.5)
                none_gt_scores = outputs_bs.masked_select(none_gt_index).reshape((outputs_bs.size(0), outputs.size(1)-1))
                #print("none_gt_index:", none_gt_index.size(), none_gt_index)
                #print("none_gt_scores:", none_gt_scores.size(), none_gt_scores)
                hard_scores = none_gt_scores.topk(K, dim=1)[0]
                #print("hard_scores:", hard_scores.size(), hard_scores)
                #the index of hard samples, i.e., samples of old classes
                hard_index = targets.lt(num_old_classes)
                hard_num = torch.nonzero(hard_index).size(0)
                #print("hard examples size: ", hard_num)
                if  hard_num > 0:
                    gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
                    hard_scores = hard_scores[hard_index]
                    assert(gt_scores.size() == hard_scores.size())
                    assert(gt_scores.size(0) == hard_num)
                    #print("hard example gt scores: ", gt_scores.size(), gt_scores)
                    #print("hard example max novel scores: ", hard_scores.size(), hard_scores)
                    loss3 = nn.MarginRankingLoss(margin=dist)(gt_scores.view(-1, 1), \
                        hard_scores.view(-1, 1), torch.ones(hard_num*K).to(device)) * lw_mr
                else:
                    loss3 = torch.zeros(1).to(device)
                #################################################
                loss = loss1 + loss2 + loss3
            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            if iteration > start_iteration:
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
                train_loss3 += loss3.item()
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
            print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f}, Train Loss3: {:.4f},\
                Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader), \
                train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), train_loss3/(batch_idx+1),
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
        handle_old_scores_bs.remove()
        handle_new_scores_bs.remove()
    return tg_model