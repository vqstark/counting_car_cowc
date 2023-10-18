import torch
import argparse
import os
import glob
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from models.model import ResCeptionNet
from models.resnet50 import ResNet
from utils.utils import hyp_parse, model_info
from models.focal_loss import FocalLoss
from dataset import *

def compute_histogram(dataset, max_car):
    hist = np.zeros(shape=[10**3, ], dtype=int)
    labels = list()
    for item in dataset:
        label = int((item['mask'][:, :] > 0).sum())
        if label > max_car:
            label = max_car
        labels.append(label)
        hist[label] += 1
    return labels, hist[:max_car + 1]

def compute_class_weight(histogram, car_max):
    histogram_new = np.empty(shape=[(car_max + 1),])

    histogram_new[:car_max] = histogram[:car_max]
    histogram_new[car_max] = histogram[car_max:].sum()

    class_weight = 1.0 / histogram_new

    return class_weight

def train(config, args, hyps):
    epochs = int(hyps['epochs'])
    batch_size = int(hyps['batch_size'])
    start_epoch = 1
    best_f1 = 0
    max_car = int(hyps['MAX_CAR'])

    train_ds = COWC(paths = args.annotation_train_path, 
                    root = args.imgs_train_path)
    
    print('Computing histogram and sample weights for train ds:')
    # Get labels from dataset
    train_labels, train_histogram = compute_histogram(train_ds, max_car)

    # Compute class weights
    train_weights = compute_class_weight(train_histogram, max_car)
    train_samples_weight = np.array([train_weights[i] for i in train_labels])
    train_samples_weight = torch.from_numpy(train_samples_weight)
    print('Histogram: ', train_histogram)
    print('Class weights: ', train_weights)
    train_sampler = WeightedRandomSampler(train_samples_weight, len(train_samples_weight))

    train_collater = Collater(crop_size=int(hyps['CROP_SIZE']),
                        transpose_image=True,
                        count_ignore_width=int(hyps['MARGIN']),
                        label_max=max_car,
                        random_crop=True,
                        random_flip=True, 
                        _random_color_distort=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=os.cpu_count() - 1,
        collate_fn = train_collater,
        # sampler=train_sampler,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    val_ds = COWC(paths = args.annotation_val_path, 
                    root = args.imgs_val_path)

    print('Computing histogram and sample weights for val ds:')
    val_labels, val_histogram = compute_histogram(val_ds, max_car)

    val_weights = compute_class_weight(val_histogram, max_car)
    val_samples_weight = np.array([val_weights[i] for i in val_labels])
    val_samples_weight = torch.from_numpy(val_samples_weight)
    print('Histogram: ', val_histogram)
    print('Class weights: ', val_weights)
    val_sampler = WeightedRandomSampler(val_samples_weight, len(val_samples_weight))
    
    val_collater = Collater(crop_size=int(hyps['CROP_SIZE']),
                        transpose_image=True,
                        count_ignore_width=int(hyps['MARGIN']),
                        label_max=max_car,
                        random_crop=True,
                        random_flip=True, 
                        _random_color_distort=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=os.cpu_count() - 1,
        collate_fn = val_collater,
        # sampler=val_sampler,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    if args.mode == 'resnet50':
        model = ResNet(num_classes = max_car).float()
    else:
        model = ResCeptionNet(num_classes = max_car).float()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=hyps['weight_decay'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=hyps['lr'], momentum=hyps['momentum'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.3]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # Class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weight(train_histogram, max_car)
        class_weights /= class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()
    
    # Loss function
    if args.use_focal_loss:
        criterion = FocalLoss(gamma=config['gamma'], weights=class_weights) if torch.is_tensor(class_weights) else FocalLoss(gamma=config['gamma'])
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights) if torch.is_tensor(class_weights) else nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    # Model infor
    model_info(model)

    for epoch in range(start_epoch,epochs+1):
        # Train process
        print(('\n' + '%10s' * 6) % ('Epoch', 'loss', 'acc', 'pre', 'rec', 'f1'))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # progress bar
        mloss = torch.zeros(1).cuda()
        mmetrics = torch.zeros(4).cuda()
        model.train()
        for i, (ni, batch) in enumerate(pbar):
            
            optimizer.zero_grad()
            img, label = batch['image'], batch['label'].long()

            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()

            one_hot = F.one_hot(label, num_classes=max_car+1).squeeze(1).float()

            output = model(img.float())
            if args.use_focal_loss:
                output = F.softmax(output, dim=1)
                
            loss = criterion(output, label.squeeze())

            # calculate gradient
            loss.backward()
            optimizer.step()

            # Compute metrics
            _, predicted = torch.max(output, 1)
            true_labels = one_hot.argmax(dim=1)

            accuracy = torch.sum(predicted == true_labels).float() / one_hot.size(0)
            f1_sc = f1_score(true_labels.cpu(), predicted.cpu(), average='weighted', zero_division=0.0)
            precision_sc = precision_score(true_labels.cpu(), predicted.cpu(), average='weighted', zero_division=0.0)
            recall_sc = recall_score(true_labels.cpu(), predicted.cpu(), average='weighted', zero_division=0.0)

            metrics = [accuracy, precision_sc, recall_sc, f1_sc]

            mloss = (mloss * i + loss.detach().cuda()) / (i + 1)
            mmetrics = (mmetrics * i + torch.tensor(metrics).cuda()) / (i+1)
            s = ('%10s' + '%10.4g'*5) % (
                  '%g/%g' % (epoch, epochs), mloss, *mmetrics)
            pbar.set_description(s)
        
        # Val process
        print(('\n' + '%10s'*5) % ('val_loss', 'val_acc', 'val_pre', 'val_rec', 'val_f1'))
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))  # progress bar
        val_mloss = torch.zeros(1).cuda()
        val_mmetrics = torch.zeros(4).cuda()
        model.eval()
        with torch.no_grad():
            for i, (ni, batch) in enumerate(pbar):
                
                img, label = batch['image'], batch['label'].long()

                if torch.cuda.is_available():
                    img, label = img.cuda(), label.cuda()

                one_hot = F.one_hot(label, num_classes=max_car+1).squeeze(1).float()

                output = model(img.float())
                if args.use_focal_loss:
                    output = F.softmax(output, dim=1)
                    
                val_loss = criterion(output, label.squeeze())

                # Compute metrics
                _, predicted = torch.max(output, 1)
                true_labels = one_hot.argmax(dim=1)

                accuracy = torch.sum(predicted == true_labels).float() / one_hot.size(0)
                f1_sc = f1_score(true_labels.cpu(), predicted.cpu(), average='weighted', zero_division=0.0)
                precision_sc = precision_score(true_labels.cpu(), predicted.cpu(), average='weighted', zero_division=0.0)
                recall_sc = recall_score(true_labels.cpu(), predicted.cpu(), average='weighted', zero_division=0.0)

                val_metrics = [accuracy, precision_sc, recall_sc, f1_sc]

                val_mloss = (val_mloss * i + val_loss.detach().cuda()) / (i+1)
                val_mmetrics = (val_mmetrics * i + torch.tensor(val_metrics).cuda()) / (i+1)
                val_s = ('%10.4g'*5) % (val_mloss, *val_mmetrics)
                pbar.set_description(val_s)

        # Update scheduler
        scheduler.step()
        
        # Checkpoint
        if val_mmetrics[-1] > best_f1:
            best_f1 = val_mmetrics[-1]

        # Create checkpoint
        checkpoint_data = {'epoch': epoch,
                    'best_f1': best_f1,
                    'model': model.module.state_dict() if type(
                    model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                    'optimizer': optimizer.state_dict()}
        
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_mloss, "metrics": val_mmetrics},
            checkpoint=checkpoint,
        )
        
        torch.cuda.empty_cache()
    
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a counter')
    parser.add_argument('--annotation_train_path', type=str, default='F:/Quan/cowc_processed/train_val/crop/train.txt')
    parser.add_argument('--imgs_train_path', type=str, default='F:/Quan/cowc_processed/train_val/crop/train')
    parser.add_argument('--annotation_val_path', type=str, default='F:/Quan/cowc_processed/train_val/crop/val.txt')
    parser.add_argument('--imgs_val_path', type=str, default='F:/Quan/cowc_processed/train_val/crop/val')
    parser.add_argument('--mode', type=str, default='resnet50')
    parser.add_argument('--use_class_weights', type=bool, default=True, help='Use class weights')
    parser.add_argument('--use_focal_loss', type=bool, default=True, help='Use focal loss')

    args = parser.parse_args()
    # hyps = hyp_parse(args.hyp)
        
    # print(args)
    # print(hyps)

    from functools import partial

    hyps = {
        "epochs": 20,
        'CROP_SIZE': 96,
        'MARGIN': 8,
        'GRID_SIZE': 2048,
        'MAX_CAR': 9,
        'batch_size': 32,
        'epochs': 15,
        'momentum': 0.9,
        'weight_decay': 0.0002
    }

    config = {
        'lr': tune.choice([1e-2, 1e-3, 1e-4, 1e-5]),
        'gamma': tune.choice([0.5, 0.4, 0.3, 0.2])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=int(hyps['epochs']),
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        partial(train, args=args, hyps=hyps),
        config = config,
        resources_per_trial={"cpu": 8, "gpu": 1},
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")