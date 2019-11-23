import cv2 as cv
from PIL import Image as pilimg
from imageTools import medianFilter
import sys

# check that two command line arguments have been given
if len(sys.argv) != 3: # (the file name counts as an argument so we actually want 3)
    print("usage: \033[1mpython median.py\033[0m \033[4msource_file\033[0m <mask size>")
    sys.exit(1)

# get path specified on the command line and open the image there
imagePath = sys.argv[1]
imageArray = cv.imread(imagePath, 0)

# check the image is valid
if imageArray is None:
    print("'%s' appears to not be an image" % imagePath)
    sys.exit(2)

# get the mask size specified on the command line
try:
    maskSize = int(sys.argv[2])
except ValueError:
    print("can't convert '%s' into an integer" % sys.argv[2])
    sys.exit(3)

# apply median filter and create image object
outputImage = pilimg.fromarray(medianFilter(imageArray, maskSize))
outputImage.show()
