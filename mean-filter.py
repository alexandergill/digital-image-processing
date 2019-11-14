import cv2 as cv
import numpy as np
from PIL import Image as pilimg
import math
import convolveWithMask as convolve

imagePath = "images/NZjers1.png"
imageArray = cv.imread(imagePath, 0)

imageDimensions = np.shape(imageArray)
outputArray = np.zeros(imageDimensions)

mask = np.ones((11,1)) / 11

outputArray = convolve.convoleWithMask(imageArray, mask=mask)

outputImage = pilimg.fromarray(outputArray)
outputImage.show()
