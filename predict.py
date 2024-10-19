from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="lokasi model")
ap.add_argument("-i", "--image", required=True, help="lokasi gambar")
args = vars(ap.parse_args())

font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 1
font_scale = 0.4
green = (0,255,0)
red = (0,0,255)
yellow = (0,127,255)

print("[INFO] loading network . . .")
model = load_model(args["model"])

prevtime = time.time()
maxVal = 0
maxIndex = 0
status = [0,0,0]
label = ""

image = cv2.imread(args["image"])
orig = image.copy()
orig = cv2.resize(orig, (200,150))
image = cv2.resize(image, (75, 75))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# status[0],status[1],status[2],status[3] = model.predict(image)[0]
status[0],status[1],status[2]= model.predict(image)[0]
print(status[0])
print(status[1])
print(status[2])
maxVal = max(status)
maxIndex = status.index(maxVal)
# (normal, imatur, matur, hipermatur) = model.predict(image)[0]
# print(str(normal) + " " + str(insipien) + " " + str(imatur) + " " + str(matur) + " " + str(hipermatur))
#print(status)
if maxIndex == 0:
	label = "Normal"
	color = green
#	print(label)
elif maxIndex == 1:
	label = "Immature Cataract"
	color = yellow
#	print(label)
elif maxIndex == 2:
	label = "Mature Cataract"
	color = red
teks = "Condition : " + label
#	print(label)
# else:
	# label = "hipermatur"
	# print(label)
print("\n\nHasil prediksi : " + label)
cv2.rectangle(orig,(0,0),(200,20),(255,255,255),-1)
cv2.putText(orig, teks, (5,15), font, font_scale, color, thickness, cv2.LINE_AA)
cv2.imshow('Res', orig)
timeproc = time.time() - prevtime
print("\n\nProcess time : " + str(timeproc) + " s")
cv2.waitKey(0)
cv2.destroyAllWindows()