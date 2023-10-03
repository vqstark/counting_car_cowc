import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet(nn.Module):
    def __init__(self, num_classes, class_weights):
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
        self.loss = nn.CrossEntropyLoss(weight = self.class_weights)
    
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
                
                true_positives = torch.sum((predicted == 1) & (true_labels == 1)).float()
                false_positives = torch.sum((predicted == 1) & (true_labels == 0)).float()
                false_negatives = torch.sum((predicted == 0) & (true_labels == 1)).float()
                
                accuracy = torch.sum(predicted == true_labels).float() / y.size(0)
                precision = true_positives / (true_positives + false_positives + 1e-15)
                recall = true_positives / (true_positives + false_negatives + 1e-15)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-15)
        return loss, accuracy, precision, recall, f1

    def predict(self, x):
        _, argmax = torch.max(x, 1)
        return argmax