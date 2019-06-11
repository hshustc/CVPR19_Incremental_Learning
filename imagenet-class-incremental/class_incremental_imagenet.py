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

import utils_pytorch
from utils_imagenet.utils_dataset import split_images_labels
from utils_imagenet.utils_dataset import merge_images_labels
from utils_incremental.compute_features import compute_features
from utils_incremental.compute_accuracy import compute_accuracy
from utils_incremental.compute_confusion_matrix import compute_confusion_matrix
from utils_incremental.incremental_train_and_eval import incremental_train_and_eval

######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='seed_1993_subset_100_imagenet', type=str)
parser.add_argument('--datadir', default='data/seed_1993_subset_100_imagenet/data', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--num_workers', default=16, type=int, \
    help='the number of workers for loading data')
parser.add_argument('--nb_cl_fg', default=50, type=int, \
    help='the number of classes in first group')
parser.add_argument('--nb_cl', default=10, type=int, \
    help='Classes per group')
parser.add_argument('--nb_protos', default=20, type=int, \
    help='Number of prototypes per class at the end')
parser.add_argument('--nb_runs', default=1, type=int, \
    help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--ckp_prefix', default=os.path.basename(sys.argv[0])[:-3], type=str, \
    help='Checkpoint prefix')
parser.add_argument('--epochs', default=90, type=int, \
    help='Epochs')
parser.add_argument('--T', default=2, type=float, \
    help='Temporature for distialltion')
parser.add_argument('--beta', default=0.25, type=float, \
    help='Beta for distialltion')
parser.add_argument('--resume', action='store_true', \
    help='resume from checkpoint')
parser.add_argument('--fix_budget', action='store_true', \
    help='fix budget')
parser.add_argument('--rs_ratio', default=0, type=float, \
    help='The ratio for resample')
parser.add_argument('--random_seed', default=1993, type=int, \
    help='random seed')
args = parser.parse_args()

########################################
assert(args.nb_cl_fg % args.nb_cl == 0)
assert(args.nb_cl_fg >= args.nb_cl)
train_batch_size       = 128            # Batch size for train
test_batch_size        = 50             # Batch size for test
eval_batch_size        = 128            # Batch size for eval
base_lr                = 0.1            # Initial learning rate
lr_strat               = [30, 60]       # Epochs where learning rate gets decreased
lr_factor              = 0.1            # Learning rate decrease factor
custom_weight_decay    = 1e-4           # Weight Decay
custom_momentum        = 0.9            # Momentum
args.ckp_prefix        = '{}_nb_cl_fg_{}_nb_cl_{}_nb_protos_{}'.format(args.ckp_prefix, args.nb_cl_fg, args.nb_cl, args.nb_protos)
np.random.seed(args.random_seed)        # Fix the random seed
print(args)
########################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#transform_train = transforms.Compose([
#    transforms.RandomCrop(32, padding=4),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),
#])
#transform_test = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),
#])
#trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
#                                        download=True, transform=transform_train)
#testset = torchvision.datasets.CIFAR100(root='./data', train=False,
#                                       download=True, transform=transform_test)
#evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
#                                       download=False, transform=transform_test)
# Data loading code
traindir = os.path.join(args.datadir, 'train')
valdir = os.path.join(args.datadir, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trainset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
testset =  datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
evalset =  datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

# Initialization
dictionary_size     = 1500
top1_acc_list_cumul = np.zeros((int(args.num_classes/args.nb_cl),3,args.nb_runs))
top1_acc_list_ori   = np.zeros((int(args.num_classes/args.nb_cl),3,args.nb_runs))

#X_train_total = np.array(trainset.train_data)
#Y_train_total = np.array(trainset.train_labels)
#X_valid_total = np.array(testset.test_data)
#Y_valid_total = np.array(testset.test_labels)
X_train_total, Y_train_total = split_images_labels(trainset.imgs)
X_valid_total, Y_valid_total = split_images_labels(testset.imgs)

# Launch the different runs
for iteration_total in range(args.nb_runs):
    # Select the order for the class learning
    order_name = "./checkpoint/seed_{}_{}_order_run_{}.pkl".format(args.random_seed, args.dataset, iteration_total)
    print("Order name:{}".format(order_name))
    if os.path.exists(order_name):
        print("Loading orders")
        order = utils_pytorch.unpickle(order_name)
    else:
        print("Generating orders")
        order = np.arange(args.num_classes)
        np.random.shuffle(order)
        utils_pytorch.savepickle(order, order_name)
    order_list = list(order)
    print(order_list)

    # Initialization of the variables for this run
    X_valid_cumuls    = []
    X_protoset_cumuls = []
    X_train_cumuls    = []
    Y_valid_cumuls    = []
    Y_protoset_cumuls = []
    Y_train_cumuls    = []
    alpha_dr_herding  = np.zeros((int(args.num_classes/args.nb_cl),dictionary_size,args.nb_cl),np.float32)

    # The following contains all the training samples of the different classes
    # because we want to compare our method with the theoretical case where all the training samples are stored
    # prototypes = np.zeros((args.num_classes,dictionary_size,X_train_total.shape[1],X_train_total.shape[2],X_train_total.shape[3]))
    prototypes = [[] for i in range(args.num_classes)]
    for orde in range(args.num_classes):
        prototypes[orde] = X_train_total[np.where(Y_train_total==order[orde])]
    prototypes = np.array(prototypes)

    start_iter = int(args.nb_cl_fg/args.nb_cl)-1
    for iteration in range(start_iter, int(args.num_classes/args.nb_cl)):
        #init model
        if iteration == start_iter:
            ############################################################
            last_iter = 0
            ############################################################
            tg_model = models.resnet18(num_classes=args.nb_cl_fg)
            ref_model = None
        else:
            ############################################################
            last_iter = iteration
            ############################################################
            #increment classes
            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            new_fc = nn.Linear(in_features, out_features+args.nb_cl)
            new_fc.weight.data[:out_features] = tg_model.fc.weight.data
            new_fc.bias.data[:out_features] = tg_model.fc.bias.data
            tg_model.fc = new_fc

        # Prepare the training data for the current batch of classes
        actual_cl        = order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)]
        indices_train_10 = np.array([i in order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_train_total])
        indices_test_10  = np.array([i in order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_valid_total])

        X_train          = X_train_total[indices_train_10]
        X_valid          = X_valid_total[indices_test_10]
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul    = np.concatenate(X_valid_cumuls)
        X_train_cumul    = np.concatenate(X_train_cumuls)

        Y_train          = Y_train_total[indices_train_10]
        Y_valid          = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul    = np.concatenate(Y_valid_cumuls)
        Y_train_cumul    = np.concatenate(Y_train_cumuls)

        # Add the stored exemplars to the training data
        if iteration == start_iter:
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            X_protoset = np.concatenate(X_protoset_cumuls)
            Y_protoset = np.concatenate(Y_protoset_cumuls)
            if args.rs_ratio > 0:
                #1/rs_ratio = (len(X_train)+len(X_protoset)*scale_factor)/(len(X_protoset)*scale_factor)
                scale_factor = (len(X_train) * args.rs_ratio) / (len(X_protoset) * (1 - args.rs_ratio))
                rs_sample_weights = np.concatenate((np.ones(len(X_train)), np.ones(len(X_protoset))*scale_factor))
                #number of samples per epoch
                #rs_num_samples = len(X_train) + len(X_protoset)
                rs_num_samples = int(len(X_train) / (1 - args.rs_ratio))
                print("X_train:{}, X_protoset:{}, rs_num_samples:{}".format(len(X_train), len(X_protoset), rs_num_samples))
            X_train    = np.concatenate((X_train,X_protoset),axis=0)
            Y_train    = np.concatenate((Y_train,Y_protoset))

        # Launch the training loop
        print('Batch of classes number {0} arrives ...'.format(iteration+1))
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        ############################################################
        #trainset.train_data = X_train.astype('uint8')
        #trainset.train_labels = map_Y_train
        current_train_imgs = merge_images_labels(X_train, map_Y_train)
        trainset.imgs = trainset.samples = current_train_imgs
        if iteration > start_iter and args.rs_ratio > 0 and scale_factor > 1:
            print("Weights from sampling:", rs_sample_weights)
            index1 = np.where(rs_sample_weights>1)[0]
            index2 = np.where(map_Y_train<iteration*args.nb_cl)[0]
            assert((index1==index2).all())
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(rs_sample_weights, rs_num_samples)
            #trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, \
            #    shuffle=False, sampler=train_sampler, num_workers=2)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, \
                shuffle=False, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)             
        else:
            #trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
            #    shuffle=True, num_workers=2)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                shuffle=True, num_workers=args.num_workers, pin_memory=True)
        #testset.test_data = X_valid_cumul.astype('uint8')
        #testset.test_labels = map_Y_valid_cumul
        current_test_images = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
        testset.imgs = testset.samples = current_test_images
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
            shuffle=False, num_workers=2)
        print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
        print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))
        ##############################################################
        ckp_name = './checkpoint/{}_run_{}_iteration_{}_model.pth'.format(args.ckp_prefix, iteration_total, iteration)
        print('ckp_name', ckp_name)
        if args.resume and os.path.exists(ckp_name):
            print("###############################")
            print("Loading models from checkpoint")
            tg_model = torch.load(ckp_name)
            print("###############################")
        else:
            tg_params = tg_model.parameters()
            tg_model = tg_model.to(device)
            if iteration > start_iter:
                ref_model = ref_model.to(device)
            tg_optimizer = optim.SGD(tg_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
            tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
            tg_model = incremental_train_and_eval(args.epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
                trainloader, testloader, \
                iteration, start_iter, \
                args.T, args.beta)
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(tg_model, ckp_name)

        ### Exemplars
        if args.fix_budget:
            nb_protos_cl = int(np.ceil(args.nb_protos*args.num_classes*1.0/args.nb_cl/(iteration+1)))
        else:
            nb_protos_cl = args.nb_protos
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        num_features = tg_model.fc.in_features
        # Herding
        print('Updating exemplar set...')
        for iter_dico in range(last_iter*args.nb_cl, (iteration+1)*args.nb_cl):
            # Possible exemplars in the feature space and projected on the L2 sphere
            # evalset.test_data = prototypes[iter_dico].astype('uint8')
            # evalset.test_labels = np.zeros(evalset.test_data.shape[0]) #zero labels
            current_eval_set = merge_images_labels(prototypes[iter_dico], np.zeros(len(prototypes[iter_dico])))
            evalset.imgs = evalset.samples = current_eval_set
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                shuffle=False, num_workers=args.num_workers, pin_memory=True)
            num_samples = len(prototypes[iter_dico])            
            mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
            D = mapped_prototypes.T
            D = D/np.linalg.norm(D,axis=0)

            # Herding procedure : ranking of the potential exemplars
            mu  = np.mean(D,axis=1)
            index1 = int(iter_dico/args.nb_cl)
            index2 = iter_dico % args.nb_cl
            alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
            w_t = mu
            iter_herding     = 0
            iter_herding_eff = 0
            while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                tmp_t   = np.dot(w_t,D)
                ind_max = np.argmax(tmp_t)
                iter_herding_eff += 1
                if alpha_dr_herding[index1,ind_max,index2] == 0:
                    alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                    iter_herding += 1
                w_t = w_t+mu-D[:,ind_max]

        # Prepare the protoset
        X_protoset_cumuls = []
        Y_protoset_cumuls = []

        # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
        print('Computing mean-of_exemplars and theoretical mean...')
        # class_means = np.zeros((64,100,2))
        class_means = np.zeros((num_features, args.num_classes, 2))
        for iteration2 in range(iteration+1):
            for iter_dico in range(args.nb_cl):
                current_cl = order[range(iteration2*args.nb_cl,(iteration2+1)*args.nb_cl)]

                # Collect data in the feature space for each class
                # evalset.test_data = prototypes[iteration2*args.nb_cl+iter_dico].astype('uint8')
                # evalset.test_labels = np.zeros(evalset.test_data.shape[0]) #zero labels
                current_eval_set = merge_images_labels(prototypes[iteration2*args.nb_cl+iter_dico], \
                    np.zeros(len(prototypes[iteration2*args.nb_cl+iter_dico])))
                evalset.imgs = evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
                num_samples = len(prototypes[iteration2*args.nb_cl+iter_dico])
                mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D,axis=0)
                # Flipped version also
                # evalset.test_data = prototypes[iteration2*args.nb_cl+iter_dico][:,:,:,::-1].astype('uint8')
                # evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                #     shuffle=False, num_workers=2)
                # mapped_prototypes2 = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                # D2 = mapped_prototypes2.T
                # D2 = D2/np.linalg.norm(D2,axis=0)
                D2 = D

                # iCaRL
                alph = alpha_dr_herding[iteration2,:,iter_dico]
                assert((alph[num_samples:]==0).all())
                alph = alph[:num_samples]
                alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                # X_protoset_cumuls.append(prototypes[iteration2*args.nb_cl+iter_dico,np.where(alph==1)[0]])
                X_protoset_cumuls.append(prototypes[iteration2*args.nb_cl+iter_dico][np.where(alph==1)[0]])
                Y_protoset_cumuls.append(order[iteration2*args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                alph = alph/np.sum(alph)
                class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])

                # Normal NCM
                # alph = np.ones(dictionary_size)/dictionary_size
                alph = np.ones(num_samples)/num_samples
                class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])

        torch.save(class_means, \
            './checkpoint/{}_run_{}_iteration_{}_class_means.pth'.format(args.ckp_prefix,iteration_total, iteration))

        current_means = class_means[:, order[range(0,(iteration+1)*args.nb_cl)]]
        ##############################################################
        # Calculate validation error of model on the first nb_cl classes:
        map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
        print('Computing accuracy on the original batch of classes...')
        # evalset.test_data = X_valid_ori.astype('uint8')
        # evalset.test_labels = map_Y_valid_ori
        current_eval_set = merge_images_labels(X_valid_ori, map_Y_valid_ori)
        evalset.imgs = evalset.samples = current_eval_set
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        ori_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader)
        top1_acc_list_ori[iteration, :, iteration_total] = np.array(ori_acc).T
        ##############################################################
        # Calculate validation error of model on the cumul of classes:
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        print('Computing cumulative accuracy...')
        # evalset.test_data = X_valid_cumul.astype('uint8')
        # evalset.test_labels = map_Y_valid_cumul
        current_eval_set = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
        evalset.imgs = evalset.samples = current_eval_set
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                shuffle=False, num_workers=args.num_workers, pin_memory=True)        
        cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader)
        top1_acc_list_cumul[iteration, :, iteration_total] = np.array(cumul_acc).T
        ##############################################################
        # Calculate confusion matrix
        print('Computing confusion matrix...')
        cm = compute_confusion_matrix(tg_model, tg_feature_model, current_means, evalloader)
        cm_name = './checkpoint/{}_run_{}_iteration_{}_confusion_matrix.pth'.format(args.ckp_prefix,iteration_total, iteration)
        with open(cm_name, 'wb') as f:
            pickle.dump(cm, f, 2) #for reading with Python 2
        ##############################################################   

    # Final save of the data
    torch.save(top1_acc_list_ori, \
        './checkpoint/{}_run_{}_top1_acc_list_ori.pth'.format(args.ckp_prefix, iteration_total))
    torch.save(top1_acc_list_cumul, \
        './checkpoint/{}_run_{}_top1_acc_list_cumul.pth'.format(args.ckp_prefix, iteration_total))
