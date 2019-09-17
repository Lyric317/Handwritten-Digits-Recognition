import numpy as np 
import math
import cv2
from scipy import ndimage

#creat an array to store the test images
images = np.zeros((1,784))
#one hot labels
#labels = np.zeros((5,10))

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

#i = 0
#for No in [9,2,5,0,7]:
	#read the image
	#gray = cv2.imread(str(No)+".jpg",cv2.IMREAD_GRAYSCALE)
	
	#threshold the image
	#(thresh, gray) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
def imreset(gray):
	label = np.zeros((1,10))

	kerneld = np.ones((1,1),np.uint8)
	gray = cv2.dilate(gray,kerneld,iterations = 3)
	#kernel = np.ones((2,2),np.uint8)
	#gray= cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

	#gray[:] = map(list,zip(*gray[::-1]))



	#resize the image and invert it
	gray = cv2.resize(255-gray,(28,28))

	while np.sum(gray[0]) == 0:
		gray = gray[1:]

	while np.sum(gray[:,0]) == 0:
		gray = np.delete(gray,0,1)

	while np.sum(gray[-1]) == 0:
		gray = gray[:-1]

	while np.sum(gray[:,-1]) == 0:
		gray = np.delete(gray,-1,1)

	rows,cols = gray.shape

	if rows > cols:
		factor = 20.0/rows
		rows = 20
		cols = int(round(cols*factor))
		gray = cv2.resize(gray, (cols,rows))
	else:
		factor = 20.0/cols
		cols = 20
		rows = int(round(rows*factor))
		gray = cv2.resize(gray, (cols, rows))

	colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
	rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
	gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

	shiftx,shifty = getBestShift(gray)
	shifted = shift(gray,shiftx,shifty)
	gray = shifted

	cv2.namedWindow("gray")
	cv2.imshow("gray",gray)
	cv2.waitKey(0)

	image = gray.flatten()
	image.resize((1,784))
	return image,label
