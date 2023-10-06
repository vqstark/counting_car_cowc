import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class ResNet(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super(ResNet, self).__init__()

        self.num_classes = num_classes + 1
        self.class_weights = class_weights

        self.resnet50 = resnet50(pretrained=False)

        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_features, 512),  # Custom fully connected layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)  # Output layer with num_classes units
        )

        self.softmax = nn.LogSoftmax(dim=1)

        if torch.is_tensor(self.class_weights):
            self.loss = nn.CrossEntropyLoss(weight = self.class_weights)
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None, training=True):
        x = self.resnet50(x) # Features map
        x = self.softmax(x)

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

                f1_sc = f1_score(true_labels.cpu(), predicted.cpu(), average='weighted', zero_division=0.0)
                precision_sc = precision_score(true_labels.cpu(), predicted.cpu(), average='weighted', zero_division=0.0)
                recall_sc = recall_score(true_labels.cpu(), predicted.cpu(), average='weighted', zero_division=0.0)
                
        return loss, accuracy, precision_sc, recall_sc, f1_sc
    
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
        return argmax