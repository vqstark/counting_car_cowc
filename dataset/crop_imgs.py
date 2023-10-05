"""
    References: https://github.com/motokimura/cowc_car_counting/blob/master/src/features/gen_train_val_crops.py
"""
import argparse
import os
import shutil
import math
import numpy as np
from PIL import Image
from skimage import io
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 1000000000

from create_processed_folder import create_nested_folder

def hyp_parse(hyp_path):
    hyp = {}
    keys = [] 
    with open(hyp_path,'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip())==0 : continue
            v = line.strip().split(':')
            try:
                hyp[v[0]] = float(v[1].strip().split(' ')[0])
            except:
                hyp[v[0]] = eval(v[1].strip().split(' ')[0])
            keys.append(v[0])
        f.close()
    return hyp

def dump_crop_filenames(out_txt, crop_filenames):

	with open(out_txt, 'w') as f:
		for i, crop_filename in enumerate(crop_filenames):
			if i != 0:
				f.write("\n")
			f.write(crop_filename)


def get_crop_centers(label_car, label_neg, seed=0):

	car_ys, car_xs = np.where(label_car > 0)
	points_car = np.concatenate([car_ys[:, None], car_xs[:, None]], axis=1)

	neg_ys, neg_xs = np.where(label_neg > 0)
	points_neg = np.concatenate([neg_ys[:, None], neg_xs[:, None]], axis=1)

	crop_centers = np.concatenate([points_car, points_neg], axis=0)

	np.random.seed(seed)
	np.random.shuffle(crop_centers)

	return crop_centers


def gen_train_area_mask(h, w, grid_size):

	yi_max, xi_max = int(math.ceil(h / grid_size)), int(math.ceil(w / grid_size))

	# 'train_area_mask' has True value at (x,y) assigned to train and False at (x,y) assigned to val.
	train_area_mask = np.ones(shape=[yi_max * grid_size, xi_max * grid_size], dtype=bool)

	for yi in range(yi_max):
		for xi in range(xi_max):

			top = yi * grid_size
			left = xi * grid_size
			bottom = top + grid_size
			right = left + grid_size

			grid_cell = np.ones(shape=[grid_size, grid_size], dtype=bool)

			# Divide a cell on the grid into 4 parts and assign botom-right one as val and others as train.
			for j in range(2):
				for i in range(2):

					top_local = j * grid_size // 2
					left_local = i * grid_size // 2
					bottom_local = top_local + grid_size // 2
					right_local = left_local + grid_size // 2

					if (j * 2 + i) == 3:
						grid_cell[top_local:bottom_local, left_local:right_local] = False

			train_area_mask[top:bottom, left:right] = grid_cell

	# Cut-off right and bottom side so that it has the same height/width with the original scene image.
	train_area_mask = train_area_mask[:h, :w]

	return train_area_mask


def gen_crops_from_scene(image, label_car, label_neg, train_area_mask, out_basename, train_dst, val_dst, tcrop_size, vcrop_size, seed):

	h, w, _ = image.shape

	crop_centers = get_crop_centers(label_car, label_neg, seed)

	train_crop_footprint = np.zeros(shape=[h, w], dtype=bool)
	val_crop_footprint = np.zeros(shape=[h, w], dtype=bool)

	train_filenames = []
	val_filenames = []

	train_centers = []
	val_centers = []

	for crop_center in tqdm(crop_centers):

		y, x = crop_center

		is_train = train_area_mask[y, x]
		crop_size = tcrop_size if is_train else vcrop_size

		top = y - crop_size // 2
		left = x - crop_size // 2
		bottom = top + crop_size
		right = left + crop_size

		if (top < 0) or (left < 0) or (bottom > h) or (right > w):
			continue

		crop_filename = "{}_{}_{}.png".format(out_basename, y, x)

		if is_train:
			# Check if the crop has overlap with val crops already extracted.
			if val_crop_footprint[top:bottom, left:right].max() == True:
				continue

			train_crop_footprint[top:bottom, left:right] = True
			train_filenames.append(crop_filename)
			train_centers.append((y, x))

		else:
			# Check if the crop has overlap with train crops already extracted.
			if train_crop_footprint[top:bottom, left:right].max() == True:
				continue

			val_crop_footprint[top:bottom, left:right] = True
			val_filenames.append(crop_filename)
			val_centers.append((y, x))

		image_crop = image[top:bottom, left:right]
		label_crop = label_car[top:bottom, left:right]

		out_image = np.empty(shape=(crop_size, crop_size * 2, 3), dtype=np.uint8)
		out_image[:, :crop_size] = image_crop
		out_image[:, crop_size:] = np.repeat(label_crop[:, :, None], 3, axis=2)

		dst_dir = train_dst if is_train else val_dst

		if dst_dir is not None:
			out_path = os.path.join(dst_dir, crop_filename)
			io.imsave(out_path, out_image)

	return train_filenames, val_filenames, train_centers, val_centers


def gen_train_val_crops(root_dir, scene_list, out_dir, tcrop_size, vcrop_size, grid_size, seed):

	train_dst  = None if (out_dir is None) else os.path.join(out_dir, "train")
	val_dst    = None if (out_dir is None) else os.path.join(out_dir, "val")
	train_list = None if (out_dir is None) else os.path.join(out_dir, "train.txt")
	val_list   = None if (out_dir is None) else os.path.join(out_dir, "val.txt")

	if train_dst is not None:
		os.makedirs(train_dst, exist_ok=True)

	if val_dst is not None:
		os.makedirs(val_dst, exist_ok=True)

	with open(scene_list) as f:
		scenes = f.readlines()

	train_filenames = []
	val_filenames = []

	for idx, scene in enumerate(scenes):
		scene = scene.rstrip()

		print("Loading {} ... ({}/{})".format(scene, idx + 1, len(scenes)))

		image_path = os.path.join(root_dir, "{}.png".format(scene))
		car_path = os.path.join(root_dir, "{}_Annotated_Cars.png".format(scene))
		neg_path = os.path.join(root_dir, "{}_Annotated_Negatives.png".format(scene))

		image = io.imread(image_path)
		image = image[:, :, :3] # remove alpha channel

		label_car = io.imread(car_path)
		label_car = label_car[:, :, 3] # use alpha channel

		label_neg = io.imread(neg_path)
		label_neg = label_neg[:, :, 3] # use alpha channel

		h, w, _ = image.shape
		train_area_mask = gen_train_area_mask(h, w, grid_size)

		out_basename, _ = os.path.splitext(os.path.basename(image_path))

		train_crops, val_crops, _, _ = gen_crops_from_scene(image, label_car, label_neg, train_area_mask, out_basename, train_dst, val_dst, tcrop_size, vcrop_size, seed)

		train_filenames.extend(train_crops)
		val_filenames.extend(val_crops)

	if train_list is not None:
		dump_crop_filenames(train_list, train_filenames)

	if val_list is not None:
		dump_crop_filenames(val_list, val_filenames)

	print("Done!")

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--root_dir', help='Root directory for cowc ground_truth_sets dir',
						default='../cowc/datasets/ground_truth_sets')
	parser.add_argument('--dest-dir', help='Root directory for cowc ground_truth_sets dir',
						default='../cowc_processed')
	parser.add_argument('--scene-list', help='Path to a text listing up source cowc image and label data',
						default='../cowc_processed/train_val/train_val_scenes.txt')
	parser.add_argument('--out-dir', '-o', help='Output directory',
						default='../cowc_processed/train_val/crop')
	parser.add_argument('--hyp', type=str, default='hyps.py', help='hyper-parameter path')
	parser.add_argument('--seed', help='Random seed to suffle train/val crops', type=int, 
						default=0)
	parser.add_argument('--mode', type=str, default='resception_net')

	args = parser.parse_args()
	hyps = hyp_parse(args.hyp)

	if hyps['CROP_SIZE'] <= 96:
		hyps['MAX_CAR'] = 9

	print(args)
	print(hyps)

	create_nested_folder(args.dest_dir)

	vcrop_size = int(hyps['CROP_SIZE'])
	tcrop_size = vcrop_size + 2 * int(hyps['MARGIN'])
	grid_size = int(hyps['GRID_SIZE'])

	gen_train_val_crops(args.root_dir, args.scene_list, args.out_dir, tcrop_size, vcrop_size, grid_size, args.seed)