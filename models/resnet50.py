import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        self.num_classes = num_classes + 1

        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)

        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_features, 256),  # Custom fully connected layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)  # Output layer with num_classes units
        )

    def forward(self, x):
        x = self.resnet50(x) # Features map
        x = F.softmax(x, dim=1)
        
        return x