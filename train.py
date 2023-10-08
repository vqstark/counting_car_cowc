import torch
import argparse
import os
import glob
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from models.model import ResCeptionNet
from models.resnet50 import ResNet
from utils.utils import hyp_parse, model_info
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

def train(args, hyps):
    epochs = int(hyps['epochs'])
    batch_size = int(hyps['batch_size'])
    start_epoch = 1
    results_file = args.results_file
    checkpoint_last = args.checkpoint_last
    checkpoint_best = args.checkpoint_best
    if args.resume:
        weight_path = checkpoint_last
    best_f1 = 0
    max_car = int(hyps['MAX_CAR'])

    # creat folder
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    for f in glob.glob(results_file):
        os.remove(f)

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
                        mean = train_ds.mean,
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
        sampler=train_sampler,
        # shuffle=True,
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
                        mean = val_ds.mean,
                        transpose_image=True,
                        count_ignore_width=int(hyps['MARGIN']),
                        label_max=max_car,
                        random_crop=False,
                        random_flip=False, 
                        _random_color_distort=False)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=os.cpu_count() - 1,
        collate_fn = val_collater,
        sampler=val_sampler,
        # shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weight(train_histogram, max_car)
        class_weights /= class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

    if args.mode == 'resnet50':
        model = ResNet(num_classes = max_car, class_weights=class_weights).float()
    else:
        model = ResCeptionNet(num_classes = max_car, class_weights=class_weights).float()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'], weight_decay=hyps['weight_decay'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=hyps['lr'], momentum=hyps['momentum'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.7, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        chkpt = torch.load(weight_path)
        # load model
        if 'model' in chkpt.keys() :
            model.load_state_dict(chkpt['model'])
        else:
            model.load_state_dict(chkpt)
        # load optimizer
        if 'optimizer' in chkpt.keys() and chkpt['optimizer'] is not None and args.resume :
            optimizer.load_state_dict(chkpt['optimizer'])
            best_f1 = chkpt['best_f1']
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        # load results
        if 'training_results' in chkpt.keys() and  chkpt.get('training_results') is not None and args.resume:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt
        if 'epoch' in chkpt.keys():
            start_epoch = chkpt['epoch'] + 1   

        del chkpt

    # Model infor
    model_info(model)

    for epoch in range(start_epoch,epochs+1):
        # Train process
        print(('\n' + '%10s' * 6) % ('Epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1'))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # progress bar
        mloss = torch.zeros(1).cuda()
        mmetrics = torch.zeros(4).cuda()
        for i, (ni, batch) in enumerate(pbar):
            
            model.train()
            optimizer.zero_grad()
            img, label = batch['image'], batch['label'].long()

            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()

            one_hot = F.one_hot(label, num_classes=max_car+1).squeeze(1)

            loss, *metrics = model(img.float(), one_hot.float())

            # if torch.isnan(loss) or torch.isinf(loss):
            #     print('WARNING: NaN or infinite loss, ending training')
            #     break
            # if bool(loss == 0):
            #     continue

            # calculate gradient
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            mloss = (mloss * i + loss.detach().cuda()) / (i + 1)
            mmetrics = (mmetrics * i + torch.tensor(metrics).cuda()) / (i+1)
            s = ('%10s' + '%10.3g'*5) % (
                  '%g/%g' % (epoch, epochs), mloss, *mmetrics)
            pbar.set_description(s)
        
        # Val process
        print(('\n' + '%10s'*5) % ('val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1'))
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))  # progress bar
        val_mloss = torch.zeros(1).cuda()
        val_mmetrics = torch.zeros(4).cuda()
        model.eval()
        with torch.no_grad():
            for i, (ni, batch) in enumerate(pbar):
                
                img, label = batch['image'], batch['label'].long()

                if torch.cuda.is_available():
                    img, label = img.cuda(), label.cuda()

                one_hot = F.one_hot(label, num_classes=max_car+1).squeeze(1)

                val_loss, *val_metrics = model(img.float(), one_hot.float())

                val_mloss = (val_mloss * i + val_loss.detach().cuda()) / (i+1)
                val_mmetrics = (val_mmetrics * i + torch.tensor(val_metrics).cuda()) / (i+1)
                val_s = ('%10.3g'*5) % (val_mloss, *val_mmetrics)
                pbar.set_description(val_s)

        # Update scheduler
        scheduler.step()
        final_epoch = epoch == epochs

        # Write result log
        with open(results_file, 'a') as f:
            f.write(s + ' ' + val_s + '\n')
        
        # Checkpoint
        if val_metrics[-1] > best_f1:
            best_f1 = val_metrics[-1]

        with open(results_file, 'r') as f:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_f1': best_f1,
                     'training_results': f.read(),
                     'model': model.module.state_dict() if type(
                        model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': None if final_epoch else optimizer.state_dict()}
        
        # Save last checkpoint
        torch.save(chkpt, checkpoint_last)

        # Save best checkpoint
        if best_f1 == val_metrics[-1]:
            torch.save(chkpt, checkpoint_best)
        
        torch.cuda.empty_cache()
    
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a counter')
    parser.add_argument('--hyp', type=str, default='hyps.py', help='hyper-parameter path')
    parser.add_argument('--annotation_train_path', type=str, default='../cowc_processed/train_val/crop/train.txt')
    parser.add_argument('--imgs_train_path', type=str, default='../cowc_processed/train_val/crop/train')
    parser.add_argument('--annotation_val_path', type=str, default='../cowc_processed/train_val/crop/val.txt')
    parser.add_argument('--imgs_val_path', type=str, default='../cowc_processed/train_val/crop/val')
    parser.add_argument('--resume', type=bool, default=True, help='Resume training')
    parser.add_argument('--results_file', type=str, default='weights/results.txt')
    parser.add_argument('--checkpoint_last', type=str, default='weights/last.pth')
    parser.add_argument('--checkpoint_best', type=str, default='weights/best.pth')
    parser.add_argument('--mode', type=str, default='resnet50')
    parser.add_argument('--use_class_weights', type=bool, default=False, help='Use class weights')

    args = parser.parse_args()
    hyps = hyp_parse(args.hyp)
        
    print(args)
    print(hyps)

    train(args, hyps)