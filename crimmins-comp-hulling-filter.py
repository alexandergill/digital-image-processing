import cv2 as cv
import numpy as np
from PIL import Image as pilimg
from tqdm import tqdm

imagePath = "images/NZjers1.png"
imageArray = cv.imread(imagePath, 0)

iterations = 6

directions = [[1,0], [1,1], [0,1], [-1,1]]

# create progress bar object
pbar = tqdm(total=100, bar_format="{percentage:.1f}%|{bar}| elapsed: {elapsed} | remaining: {remaining}")
# get size of steps as a percentage
stepSize = 100 / (iterations
                  * len(directions) # number of directions
                  * np.product(imageArray.shape[0] - 2) # number of rows ignoring edges
                  * 8) # 8 steps per iteration

# loop for number of iterations
for iteration in range(iterations):

    #--- dark pixel adjustment ---
    # loop over the directions specified
    for direction in directions:

        #--- STEP 1: if a >= b+2, then increment b ---
        # loop over image in 2D, ignoring 1 pixel at each edge
        for i, row in enumerate(imageArray[1:-1], start=1):
            for j, pixel in enumerate(row[1:-1], start=1):

                # get value of previous pixel in this direction
                a = imageArray[tuple(np.subtract([i,j], direction))]
                
                # apply rule
                if a >= (pixel + 2):
                    imageArray[i,j] += 1

            # update progress bar each row
            pbar.update(stepSize)

        #--- STEP 2: if a > b and b <= c, then increment b
        # loop over image as before
        for i, row in enumerate(imageArray[1:-1], start=1):
            for j, pixel in enumerate(row[1:-1], start=1):

                # get values of previous and next pixels in this
                # direction
                a = imageArray[tuple(np.subtract([i,j], direction))]
                c = imageArray[tuple(np.add([i,j], direction))]

                # apply rule
                if a > pixel and pixel <= c:
                    imageArray[i,j] += 1

            # update progress bar each row
            pbar.update(stepSize)

        #--- STEP 3: if c > b and b <= a then increment b
        # loop as before
        for i, row in enumerate(imageArray[1:-1], start=1):
            for j, pixel in enumerate(row[1:-1], start=1):

                # get a and c pixel values as before
                a = imageArray[tuple(np.subtract([i,j], direction))]
                c = imageArray[tuple(np.add([i,j], direction))]

                # apply rule
                if c > pixel and pixel <= a:
                    imageArray[i,j] += 1

            # update progress bar each row
            pbar.update(stepSize)

        #--- STEP 4: if c >= b+2 then increment b
        for i, row in enumerate(imageArray[1:-1], start=1):
            for j, pixel in enumerate(row[1:-1], start=1):

                # get c as before
                c = imageArray[tuple(np.add([i,j], direction))]

                # apply rule
                if c >= (pixel + 2):
                    imageArray[i,j] += 1

            # update progress bar each row
            pbar.update(stepSize)

    #--- light pixel adjustment
    # loop over the directions specified
    for direction in directions:

        #--- STEP 1: if a <= b-2 then decrement b
        # loop over image
        for i, row in enumerate(imageArray[1:-1], start=1):
            for j, pixel in enumerate(row[1:-1], start=1):

                # get a value
                a = imageArray[tuple(np.subtract([i,j], direction))]

                # apply rule
                if a <= (pixel-2):
                    imageArray[i,j] -= 1

            # update progress bar each row
            pbar.update(stepSize)
        
        #--- STEP 2: if a < b and b >= c then decrement b
        for i, row in enumerate(imageArray[1:-1], start=1):
            for j, pixel in enumerate(row[1:-1], start=1):

                # get a and c
                a = imageArray[tuple(np.subtract([i,j], direction))]
                c = imageArray[tuple(np.add([i,j], direction))]

                # apply rule
                if a < pixel and pixel >= c:
                    imageArray[i,j] -= 1

            # update progress bar each row
            pbar.update(stepSize)

        #--- STEP 3: if c < b and b >= a then decrement b
        for i, row in enumerate(imageArray[1:-1], start=1):
            for j, row in enumerate(row[1:-1], start=1):

                # get a and c
                a = imageArray[tuple(np.subtract([i,j], direction))]
                c = imageArray[tuple(np.add([i,j], direction))]

                # apply rule
                if c < pixel and pixel >= a:
                    imageArray[i,j] -= 1

            # update progress bar each row
            pbar.update(stepSize)

        #--- STEP 4: if c <= b-2 then decrement b
        for i, row in enumerate(imageArray[1:-1], start=1):
            for j, pixel in enumerate(row[1:-1], start=1):

                # get c value as before
                c = imageArray[tuple(np.add([i,j], direction))]

                # apply rule
                if c <= (pixel-2):
                    imageArray[i,j] -= 1

            # update progress bar each row
            pbar.update(stepSize)

pbar.close()

outputImage = pilimg.fromarray(imageArray[1:-1,1:-1])
outputImage.show()
