import os
import math
import cv2
import numpy as np
from skimage import io
import pickle
import seaborn as sns
import itertools
from tqdm import tqdm

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

import matplotlib.pyplot as plt

from models.counter import Counting_Car_Model

def get_color_map(sns_palette):
    
    color_map = np.empty(shape=[0, 3], dtype=np.uint8)

    for color in sns_palette:
        r = int(color[0] * 255)
        g = int(color[1] * 255)
        b = int(color[2] * 255)
        rgb_byte = np.array([[r, g, b]], dtype=np.uint8)
        color_map = np.append(color_map, rgb_byte, axis=0)
    
    return color_map


def overlay_heatmap(
    cars, background_image, car_max, grid_size, cmap, 
    line_rgb=[0, 0, 0], line_thickness=2, alpha=0.5, min_car_to_show=1, background_rgb=[0, 0, 0]):
    
    yi_max, xi_max = cars.shape

    result = background_image.copy()
    heatmap = background_image.copy()

    sns_palette = sns.color_palette(cmap, n_colors=car_max + 1)
    color_map = get_color_map(sns_palette)
    
    for yi in range(yi_max):
        for xi in range(xi_max):
            
            top, left = yi * grid_size, xi * grid_size
            bottom, right = top + grid_size, left + grid_size
            
            cars_counted = cars[yi, xi]

            if cars_counted < min_car_to_show:
                if background_rgb is not None:
                    heatmap[top:bottom, left:right] = np.array(background_rgb)
            else:
                heatmap[top:bottom, left:right] = color_map[cars_counted]

                if line_thickness > 0:
                    cv2.rectangle(heatmap, (left, top), (right, bottom), line_rgb, thickness=line_thickness)
    
    cv2.addWeighted(heatmap, alpha, result, 1 - alpha, 0, result)

    return result

import cv2
import numpy as np
import itertools
import math

def plot_counts_on_heatmap(heatmap_overlayed, aoi_tblr, cars, grid_size, min_car_to_show=1):
    
    top, bottom, left, right = aoi_tblr
    
    yi_min, xi_min = int(math.floor(top / grid_size)), int(math.floor(left / grid_size))
    yi_max, xi_max = int(math.ceil(bottom / grid_size)), int(math.ceil(right / grid_size))
    
    top, left, bottom, right = yi_min * grid_size, xi_min * grid_size, yi_max * grid_size, xi_max * grid_size
    
    # Create a copy of the heatmap image to avoid modifying the original
    heatmap_with_text = heatmap_overlayed.copy()
    
    for (yi, xi) in itertools.product(range(yi_min, yi_max), range(xi_min, xi_max)):
        car_num = cars[yi, xi]
        if car_num < min_car_to_show:
            continue
        
        # Calculate the position to write the text
        x = int((xi + 0.5) * grid_size - left)
        y = int((yi + 0.5) * grid_size - top)
        
        # Define the text settings
        text = str(car_num)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # Black color in BGR format
        thickness = 2

        # Calculate the position for the text so that it's centered
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x -= text_width // 2
        y += text_height // 2
        
        # Write the text on the heatmap image
        cv2.putText(heatmap_with_text, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    
    return heatmap_with_text


def demo(model, img, min_car):
    car_max = 9
    cars, grid_size = model.count_on_scene(img, None)
    cars_counted, _ = cars
    cars_counted = np.array(cars_counted)

    heatmap_overlayed = overlay_heatmap(cars_counted, img, car_max, grid_size, cmap='viridis', line_thickness=-1, min_car_to_show=min_car)
    top, bottom, left, right = 0, img.shape[0], 0, img.shape[1]
    heatmap_with_text = plot_counts_on_heatmap(heatmap_overlayed, (top, bottom, left, right), cars_counted, grid_size, min_car_to_show=min_car)

    return heatmap_overlayed, heatmap_with_text

if __name__ == '__main__':
    demo()