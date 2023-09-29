import random, cv2
import numpy as np

try:
	from PIL import Image
	available = True
except ImportError as e:
	available = False
	_import_error = e


def _check_pillow_availability():
	if not available:
		raise ImportError('PIL cannot be loaded. Install Pillow!\n'
						  'The actual import error is as follows:\n' +
						  str(_import_error))


def _read_image_as_array(path, dtype):
	f = Image.open(path)
	try:
		image = np.asarray(f, dtype=dtype)
	finally:
		# Only pillow >= 3.0 has 'close' method
		if hasattr(f, 'close'):
			f.close()
	return image

def random_color_distort(
		img,
		brightness_delta=32,
		contrast_low=0.5, contrast_high=1.5,
		saturation_low=0.5, saturation_high=1.5,
		hue_delta=18):

	cv_img = img[::-1].astype(np.uint8) # RGB to BGR

	def convert(img, alpha=1, beta=0):
		img = img.astype(float) * alpha + beta
		img[img < 0] = 0
		img[img > 255] = 255
		return img.astype(np.uint8)

	def brightness(cv_img, delta):
		if random.randrange(2):
			return convert(
				cv_img,
				beta=random.uniform(-delta, delta))
		else:
			return cv_img

	def contrast(cv_img, low, high):
		if random.randrange(2):
			return convert(
				cv_img,
				alpha=random.uniform(low, high))
		else:
			return cv_img

	def saturation(cv_img, low, high):
		if random.randrange(2):
			cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
			cv_img[:, :, 1] = convert(
				cv_img[:, :, 1],
				alpha=random.uniform(low, high))
			return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
		else:
			return cv_img

	def hue(cv_img, delta):
		if random.randrange(2):
			cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
			cv_img[:, :, 0] = (
				cv_img[:, :, 0].astype(int) +
				random.randint(-delta, delta)) % 180
			return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
		else:
			return cv_img

	cv_img = brightness(cv_img, brightness_delta)

	if random.randrange(2):
		cv_img = contrast(cv_img, contrast_low, contrast_high)
		cv_img = saturation(cv_img, saturation_low, saturation_high)
		cv_img = hue(cv_img, hue_delta)
	else:
		cv_img = saturation(cv_img, saturation_low, saturation_high)
		cv_img = hue(cv_img, hue_delta)
		cv_img = contrast(cv_img, contrast_low, contrast_high)

	return cv_img[::-1]  # RGB to BGR