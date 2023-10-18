import torch
import numpy as np
from tqdm import tqdm
import math
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from models.resnet50 import ResNet

class Counting_Car_Model:
    def __init__(self, model_path, max_car=9, crop_size = 96, batch=8):
        self.model_path = model_path
        self.max_car = max_car
        self.crop_size = crop_size
        self.batch = batch

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([self.normalize])
        
        if self.crop_size <= 96:
            self.max_car = 9
        
        self.model = ResNet(num_classes = self.max_car).float().cuda()
        
        try:
            chkpt = torch.load(self.model_path)
            # load model
            if 'model' in chkpt.keys() :
                self.model.load_state_dict(chkpt['model'])
            print('Loading weights. Done!')
        except Exception as e:
            print(e)
            print('Use pretrain or fine tuning model')

    
    def count(self, img):
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float().cuda()
        img = self.trans(img)
        result = self.model(img).cpu()
        result = F.softmax(result, dim=1)
        return torch.max(result, dim=1)[1]
    
    def batch_count(self, batch):
        batch_size = len(batch)
        batch_img = torch.zeros(batch_size, 3, self.crop_size, self.crop_size)
        for i in range(batch_size):
            img = batch[i]
            img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
            img = self.trans(img)
            batch_img[i, :, :, :] = img

        result = self.model(batch_img.cuda()).cpu()
        result = F.softmax(result, dim=1)
        return torch.max(result, dim=1)[1]
    
    def count_on_scene(self, scene_img, scene_labels, exclude_margin = 8):
        
        w, h, _ = scene_img.shape
        
        grid_size = self.crop_size - 2*exclude_margin
        
        yi_max, xi_max = int(math.ceil(h / grid_size)), int(math.ceil(w / grid_size))
        
        h_grid, w_grid = yi_max * grid_size, xi_max * grid_size
        
        h_pad, w_pad = h_grid + 2*exclude_margin, w_grid + 2*exclude_margin
        
        scene_image_pad = 127 * np.ones(shape=[h_pad, w_pad, 3], dtype=np.uint8)
        scene_image_pad[exclude_margin:exclude_margin+h, exclude_margin:exclude_margin+w] = scene_img
        
        if scene_labels is not None:
            scene_label_pad = np.zeros(shape=[h_pad, w_pad], dtype=np.uint8)
            scene_label_pad[exclude_margin:exclude_margin+h, exclude_margin:exclude_margin+w] = scene_labels
        
        # Count cars in each tile on the grid
        cars_counted = np.zeros(shape=[yi_max, xi_max], dtype=int)
        
        if scene_labels is not None:
            cars_labeled = np.zeros(shape=[yi_max, xi_max], dtype=int)
        
        tile_idx = 0
        for yi in tqdm(range(yi_max)):
            batch_img = []
            num_batch = 0
            top = yi * grid_size
            for xi in range(xi_max):
                left = xi * grid_size

                tile_image = scene_image_pad[top:top+self.crop_size, left:left+self.crop_size]
                
                batch_img.append(tile_image)
                # pred = self.count(tile_image)
                # cars_counted[yi, xi] = pred

                if len(batch_img) == self.batch:
                    num_batch += 1
                    cars_counted[yi, (num_batch-1)*self.batch:num_batch*self.batch] = self.batch_count(batch_img).tolist()
                    batch_img = []

                if scene_labels is not None:
                    tile_label = scene_label_pad[top:top+self.crop_size, left:left+self.crop_size]
                    label = (tile_label[exclude_margin:-exclude_margin, exclude_margin:-exclude_margin] > 0).sum()
                    cars_labeled[yi, xi] = label

                tile_idx += 1

            if len(batch_img) > 0:
                cars_counted[yi, -len(batch_img):] = self.batch_count(batch_img).tolist()
        
        if scene_labels is not None:
            return (cars_counted, cars_labeled), grid_size

        else:
            return (cars_counted, None), grid_size
        
    def read_image_as_array(self, path, dtype):
        f = Image.open(path)
        try:
            image = np.asarray(f, dtype=dtype)
        finally:
            # Only pillow >= 3.0 has 'close' method
            if hasattr(f, 'close'):
                f.close()
        return image