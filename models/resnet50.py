import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        self.num_classes = num_classes + 1

        self.resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        x = self.resnet50(x) # Features map
        return x