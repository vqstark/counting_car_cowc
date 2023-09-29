import torch
import torch.nn as nn
import math

class ResCeption(nn.Module):
  def __init__(self, in_channels, out_channels):
    """
      Args:
            in_channels - n_dims of previous layer
            out_channel: shape = (o1, (o2,o3), (o4,o5), o6)
    """
    super(ResCeption, self).__init__()

    self.reduce = nn.Conv2d(in_channels, out_channels[0], 1, 1, 0)

    self.block1 = nn.ModuleList()
    self.block1.append(nn.Conv2d(in_channels, out_channels[1][0], 1, 1, 0))
    self.block1.append(nn.BatchNorm2d(out_channels[1][0], eps=1e-5,momentum=0.9))
    self.block1.append(nn.ReLU(inplace=True))
    self.block1.append(nn.Conv2d(out_channels[1][0], out_channels[1][1], 3, 1, 1))

    self.block2 = nn.ModuleList()
    self.block2.append(nn.Conv2d(in_channels, out_channels[2][0], 1, 1, 0))
    self.block2.append(nn.BatchNorm2d(out_channels[2][0], eps=1e-5,momentum=0.9))
    self.block2.append(nn.ReLU(inplace=True))
    self.block2.append(nn.Conv2d(out_channels[2][0], out_channels[2][1], 5, 1, 2))

    self.block3 = nn.ModuleList()
    self.block3.append(nn.MaxPool2d(3, 1, 1))
    self.block3.append(nn.Conv2d(in_channels, out_channels[3], 1, 1, 0))

    self.bn = nn.BatchNorm2d(out_channels[0], eps=1e-5,momentum=0.9)
    self.relu = nn.ReLU()

    self.init_weights()

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))

  def forward(self, x):
    x_reduce, x_block1, x_block2, x_block3 = torch.clone(x), torch.clone(x), torch.clone(x), torch.clone(x)

    x_reduce = self.reduce(x_reduce)
    for layer in self.block1:
      x_block1 = layer(x_block1)
    for layer in self.block2:
      x_block2 = layer(x_block2)
    for layer in self.block3:
      x_block3 = layer(x_block3)

    concatenate = torch.cat((x_block1, x_block2, x_block3), 1)
    add = x_reduce + concatenate

    output = self.bn(add)
    output = self.relu(output)
    return output