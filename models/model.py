import torch.nn as nn
import math
from models.layers import ResCeption

class ResCeptionNet(nn.Module):
  def __init__(self, num_classes):
    super(ResCeptionNet, self).__init__()

    input_channels = 3
    self.num_classes = num_classes + 1

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
    return x