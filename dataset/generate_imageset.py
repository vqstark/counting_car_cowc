import os
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm

from dataset.utils import _read_image_as_array

class COWC(data.Dataset):
    def __init__(self, paths, root):
        
        with open(paths) as paths_file:
            self.paths = [path.rstrip() for path in paths_file]
        
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.paths[index])
        image_mask_pair = _read_image_as_array(path, np.float64)

        _, W, _ = image_mask_pair.shape

        image = image_mask_pair[:, :W//2,  :]
        mask = image_mask_pair[:,  W//2:, 0]
        
        return {'image': image, 'mask': mask}