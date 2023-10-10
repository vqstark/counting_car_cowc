import torch
import torch.nn as nn
import math
import numpy as np
from models.layers import ResCeption
import torch.nn.functional as F

class ResCeptionNet(nn.Module):
  def __init__(self, num_classes, class_weights=None):
    super(ResCeptionNet, self).__init__()

    input_channels = 3
    self.num_classes = num_classes + 1
    self.class_weights = class_weights

    self.ff = nn.ModuleList()
    # Block 1
    self.ff.append(nn.Conv2d(input_channels, 64, 7, 2, 3))
    self.ff.append(nn.BatchNorm2d(64, eps=1e-5,momentum=0.9))
    self.ff.append(nn.ReLU(inplace=True))
    self.ff.append(nn.MaxPool2d(3, 2, 0))
    # Block 2
    self.ff.append(nn.Conv2d(64, 64, 1, 1, 0))
    self.ff.append(nn.BatchNorm2d(64, eps=1e-5,momentum=0.9))
    self.ff.append(nn.ReLU(inplace=True))

    self.ff.append(nn.Conv2d(64, 192, 3, 1, 1))
    self.ff.append(nn.BatchNorm2d(192, eps=1e-5,momentum=0.9))
    self.ff.append(nn.ReLU(inplace=True))
    self.ff.append(nn.MaxPool2d(3, 2, 0))
    # Block 3
    self.ff.append(ResCeption(in_channels=192, out_channels=(192, (96, 128), (16, 32), 32)))
    self.ff.append(ResCeption(in_channels=192, out_channels=(352, (128, 192), (32, 96), 64)))
    self.ff.append(nn.MaxPool2d(3, 2, 0))
    # Block 4
    self.ff.append(ResCeption(in_channels=352, out_channels=(320, (96, 208), (16, 48), 64)))
    self.ff.append(ResCeption(in_channels=320, out_channels=(352, (112, 224), (24, 64), 64)))
    self.ff.append(ResCeption(in_channels=352, out_channels=(384, (128, 256), (24, 64), 64)))
    self.ff.append(ResCeption(in_channels=384, out_channels=(416, (144, 288), (32, 64), 64)))
    self.ff.append(ResCeption(in_channels=416, out_channels=(512, (160, 320), (32, 64), 128)))
    self.ff.append(nn.MaxPool2d(3, 2, 0))
    # Block 5
    self.ff.append(ResCeption(in_channels=512, out_channels=(576, (160, 320), (32, 128), 128)))
    self.ff.append(ResCeption(in_channels=576, out_channels=(640, (192, 384), (48, 128), 128)))
    # Block 6
    self.ff.append(nn.AvgPool2d(5, 3, 0))
    self.ff.append(nn.Conv2d(640, 128, 1, 1, 0))
    self.ff.append(nn.BatchNorm2d(128, eps=1e-5,momentum=0.9))
    self.ff.append(nn.ReLU(inplace=True))

    self.ff.append(nn.AdaptiveAvgPool2d(1))

    self.fc = nn.Linear(128, self.num_classes)
    # self.softmax = nn.LogSoftmax(dim=1)
    
    if torch.is_tensor(self.class_weights):
        self.loss = nn.CrossEntropyLoss(weight = self.class_weights)
    else:
        self.loss = nn.CrossEntropyLoss()

    self.init_weights()

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))

  def forward(self, x, y=None, training=True):
    for m in self.ff:
      x = m(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    x = F.softmax(x, dim=1)

    if not training:
      return self.predict(x)
    
    loss = self.loss(x, y)

    # Compute accuracy if y is provided
    accuracy = None
    if y is not None:
        with torch.no_grad():
            _, predicted = torch.max(x, 1)
            true_labels = y.argmax(dim=1)
            
            accuracy = torch.sum(predicted == true_labels).float() / y.size(0)
            precision, recall, f1 = self.adding_score(predicted, true_labels)
              
    return loss, accuracy, precision, recall, f1
  
  def adding_score(self, predicted, true_labels):
        precision_per_classes = list()
        recall_per_classes = list()
        f1_per_classes = list()

        predicted = predicted.cpu()
        true_labels = true_labels.cpu()

        for i in range(self.num_classes):
            true_positives = torch.sum((predicted == i) & (true_labels == i)).float()
            false_positives = torch.sum((predicted == i) & (true_labels != i)).float()
            false_negatives = torch.sum((predicted != i) & (true_labels == i)).float()

            precision_i = true_positives / (true_positives + false_positives + 1e-15)
            recall_i = true_positives / (true_positives + false_negatives + 1e-15)
            f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i + 1e-15)

            precision_per_classes.append(precision_i)
            recall_per_classes.append(recall_i)
            f1_per_classes.append(f1_i)

        precision_per_classes_tensor = torch.tensor(precision_per_classes).cuda()
        recall_per_classes_tensor = torch.tensor(recall_per_classes).cuda()
        f1_per_classes_tensor = torch.tensor(f1_per_classes).cuda()

        return torch.mean(precision_per_classes_tensor), torch.mean(recall_per_classes_tensor), torch.mean(f1_per_classes_tensor)


  def predict(self, x):
      _, argmax = torch.max(x, 1)
      return x