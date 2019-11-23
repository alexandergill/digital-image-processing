import cv2 as cv
from PIL import Image as pilimg
from imageTools import crimmins

imagePath = "images/NZjers1.png"
imageArray = cv.imread(imagePath, 0)

outputArray = crimmins(imageArray, iterations=1)

outputImage = pilimg.fromarray(outputArray[1:-1,1:-1])
outputImage.show()
