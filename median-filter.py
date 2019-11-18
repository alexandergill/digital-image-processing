import cv2 as cv
from tqdm import tqdm
import numpy as np
from PIL import Image as pilimg
import math

imagePath = "images/NZjers1.png"
imageArray = cv.imread(imagePath, 0)

maskSize = 3
cent = math.ceil((maskSize-1)/2)
print(cent)

outputArray = np.zeros(imageArray.shape)

for i, row in enumerate(tqdm(imageArray[cent:-cent]),
                        start=cent):
    for j, _pixel in enumerate(row[cent:-cent], start=cent):

        # create empty mask
        mask = np.zeros((maskSize,maskSize))

        #fill the mask with the pixel values
        for m in range(-cent,maskSize-cent):
            for n in range(-cent,maskSize-cent):
                # get the pixel values around each pixel
                mask[m+cent,n+cent] = imageArray[i+m,j+n]

        if j == 20 and i == 20:
            print(mask)
        # get the median of the mask
        outputArray[i,j] = np.median(mask)

pilimg.fromarray(outputArray).show()
