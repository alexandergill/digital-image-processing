import cv2 as cv
from tqdm import tqdm
import numpy as np
from PIL import Image as pilimg
import math

imagePath = "images/foetus.png"
imageArray = cv.imread(imagePath, 0)

maskSize = 7
cent = math.ceil((maskSize-1)/2)

from convolveWithMask import medianFilter as med

outputArray = med(imageArray, 9)

pilimg.fromarray(outputArray).show()
