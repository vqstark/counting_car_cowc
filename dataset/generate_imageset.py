import os
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from dataset.utils import _read_image_as_array, random_color_distort

class COWC(data.Dataset):
    def __init__(self, paths, root, crop_size=96, transpose_image=False,
                return_mask=False, count_ignore_width=8,
                label_max=10*8, random_crop=False,
                random_flip=False, _random_color_distort=False):
        
        with open(paths) as paths_file:
            self.paths = [path.rstrip() for path in paths_file]
        
        self.root = root
        self.crop_size = crop_size
        self.transpose_image = transpose_image
        self.return_mask = return_mask
        self.count_ignore_width = count_ignore_width
        self.label_max = label_max
        self.random_crop = random_crop
        self.random_flip = random_flip
        self._random_color_distort = _random_color_distort
        self.mean = self.compute_mean()
        # self.mean = (128.00378071846347, 122.55737532912964, 118.57989341125801)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.paths[index])
        image_mask_pair = _read_image_as_array(path, np.float64)

        _, W, _ = image_mask_pair.shape

        image = image_mask_pair[:, :W//2,  :]
        mask = image_mask_pair[:,  W//2:, 0]

        # Crop image and mask
        h, w, _ = image.shape

        if self.random_crop:
            # Random crop
            top  = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
        else:
            # Center crop
            top  = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2

        bottom = top + self.crop_size
        right = left + self.crop_size

        image = image[top:bottom, left:right]
        mask = mask[top:bottom, left:right]

        if self.random_flip:
            # Horizontal flip
            if random.randint(0, 1):
                image = image[:, ::-1, :]
                mask = mask[:, ::-1]

            # Vertical flip
            if random.randint(0, 1):
                image = image[::-1, :, :]
                mask = mask[::-1, :]

        if self._random_color_distort:
            # Apply random color distort
            image = random_color_distort(image)
            image = np.asarray(image, dtype=np.float64)

        # Normalize
        image = (image - self.mean) / 255.0

        # Remove car annotation outside the valid area
        ignore = self.count_ignore_width
        label = (mask[ignore:-ignore, ignore:-ignore] > 0).sum()
        if ignore == 0:
            label = (mask[:, :] > 0).sum()

        if label > self.label_max:
            label = self.label_max

        # Transpose image from [h, w, c] to [c, h, w]
        if self.transpose_image:
            image = image.transpose(2, 0, 1)

        if self.return_mask:
            return {'image': image, 'label': int(label), 'mask': mask}
        else:
            return {'image': image, 'label': int(label)}
    
    def compute_mean(self):
        print('Computing mean image...')

        sum_rgb = np.zeros(shape=[3, ])
        N = len(self.paths)
        for index in range(N):
            path = os.path.join(self.root, self.paths[index])
            image_mask_pair = _read_image_as_array(path, np.float64)
            _, W, _ = image_mask_pair.shape

            image = image_mask_pair[:, :W//2,  :]
            sum_rgb += image.mean(axis=(0, 1), keepdims=False)

        mean = sum_rgb / N

        print("Computing done!")
        print("Computed mean: (R, G, B) = ({}, {}, {})".format(mean[0], mean[1], mean[2]))

        return mean