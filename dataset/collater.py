import random
import numpy as np
import torch
from torchvision import transforms

from dataset.utils import random_color_distort

class Collater(object):
    def __init__(self, crop_size=96, transpose_image=True, count_ignore_width=8, 
                 random_crop=True, random_flip=True, _random_color_distort=True, label_max=9):
        self.crop_size = crop_size
        self.transpose_image = transpose_image
        self.count_ignore_width = count_ignore_width
        self.random_crop = random_crop
        self.random_flip = random_flip
        self._random_color_distort = _random_color_distort
        self.label_max = label_max
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([self.normalize])

    def __call__(self, batch):
        images = [sample['image'] for sample in batch]
        masks = [sample['mask'] for sample in batch]
        batch_size = len(batch)

        return_batch_imgs = torch.zeros(batch_size, 3, self.crop_size, self.crop_size)
        return_batch_labels = torch.zeros(batch_size, 1)

        for i in range(batch_size):
            image = images[i]
            mask = masks[i]

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

            # Remove car annotation outside the valid area
            ignore = self.count_ignore_width
            label = (mask[ignore:-ignore, ignore:-ignore] > 0).sum()

            if ignore == 0:
                label = (mask > 0).sum()

            if label > self.label_max:
                label = self.label_max

            # Transpose image from [h, w, c] to [c, h, w]
            if self.transpose_image:
                image = image.transpose(2, 0, 1)
            
            # Normalize
            image = self.trans(torch.from_numpy(image))
            
            return_batch_imgs[i, :, :, :] = image
            return_batch_labels[i, :] = int(label)

        return {'image': return_batch_imgs, 
                'label': return_batch_labels}