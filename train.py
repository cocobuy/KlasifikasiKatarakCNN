import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from LeNet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="alamat lokasi dataset")
ap.add_argument("-m", "--model", required=True, help="alamat lokasi model akan disimpan")
ap.add_argument("-p", "--plot", type=str, default="plot.png",help="nama plot yang akan dihasilkan")
args = vars(ap.parse_args())

EPOCHS = 75
INIT_LR = 0.2e-4
BS = 32

print("[INFO] loading images . . .")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image,(75, 75))
	image = img_to_array(image)
	data.append(image)
	
	label = imagePath.split(os.path.sep)[-2]
	if label == "imatur":
		label = 1
	elif label == "matur":
		label = 2
	else:
		label = 0

	labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.4, random_state=42)

trainY = to_categorical(trainY, num_classes=3)
testY = to_categorical(testY, num_classes=3)

print("[INFO] compiling model . . .")
model = LeNet.build(width=75, height=75, depth=3, classes=3)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network . . .")
H = model.fit(trainX, trainY, validation_data=(testX,testY), epochs=EPOCHS, batch_size=32)

print("[INFO] serializing network . . .")
model.save(args["model"], save_format="h5")

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])