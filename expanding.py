import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="lokasi input")
ap.add_argument("-c", "--caption", required=True, help="caption data")
ap.add_argument("-o", "--output", required=True, help="lokasi output")
ap.add_argument("-t", "--total", type=int, default=10, help="jumlah sampel output")
args = vars(ap.parse_args())

print("[INFO] loading example image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)

# aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode="nearest")
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.15, horizontal_flip=True, fill_mode="nearest")
total = 0

print("[INFO] generating images")
imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"], save_prefix=args["caption"], save_format="jpg")

for image in imageGen:
	total += 1
	
	if total == args["total"]:
		break

print("[INFO] done!")