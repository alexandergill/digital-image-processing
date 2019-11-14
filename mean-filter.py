import cv2 as cv
import numpy as np
from PIL import Image as pilimg
import math

imagePath = "images/NZjers1.png"
imageArray = cv.imread(imagePath, 0)

imageDimensions = np.shape(imageArray)
outputArray = np.zeros(imageDimensions)

maskSize = 20
mask = np.ones((maskSize,maskSize)) / maskSize**2
maskCentre = math.ceil(maskSize/2)

for i, row in enumerate(imageArray[maskCentre:-maskCentre], start=maskCentre):
    for j, column in enumerate(row[maskCentre:-maskCentre], start=maskCentre):
        for m, maskRow in enumerate(mask, start=(maskCentre-maskSize)):
            for n, maskValue in enumerate(maskRow, start=(maskCentre-maskSize)):
                outputArray[i, j] += imageArray[i + m, j + n] * mask[m+maskSize-maskCentre, n+maskSize-maskCentre]

outputArray = outputArray.astype(int)

outputImage = pilimg.fromarray(outputArray)
outputImage.show()
