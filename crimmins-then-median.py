import cv2 as cv
from PIL import Image as pilimg
import imageTools as img

imageArray = cv.imread("images/NZjers1.png", 0)

crimminsed = img.crimmins(imageArray, 4)
outputArray = img.medianFilter(crimminsed, 3)

outputImage = pilimg.fromarray(outputArray[1:-1,1:-1])
outputImage.show()
