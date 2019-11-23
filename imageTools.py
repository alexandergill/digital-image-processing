import math
import numpy as np
from tqdm import tqdm

def convolveWithMask(inputArray, mask):

    # check that mask is one- or two-dimensional
    if len(mask.shape) > 2:
        raise ValueError("mask has more than two dimensions")

    # check that mask is odd-numbered in both dimensions
    for dimension in mask.shape:
        if (dimension % 2) == 0:
            raise ValueError("mask is not odd in both dimensions")

    # get mask dimensions
    [maskRows, maskColumns] = mask.shape

    # find horizonal and vertical centre of the mask
    maskVerticalCentre = math.ceil(maskRows/2)
    maskHorizontalCentre = math.ceil(maskColumns/2)

    # create empty output array of same shape as input
    outputArray = np.zeros(inputArray.shape)

    print('convolving with mask...')

    # loop over the rows in the image, ignoring half the mask at the
    # top and the bottom. 'tqdm' adds a progress bar. 'start' gives
    # the correct index within the loop.
    for i, row in enumerate(tqdm(inputArray[maskVerticalCentre:-maskVerticalCentre]),
                            start=maskVerticalCentre):

        # loop over the pixels in the same way
        for j, _pixel in enumerate(row[maskHorizontalCentre:-maskHorizontalCentre],
                                    start=maskHorizontalCentre):

            # loop over the rows of the mask. 'start' is set so that
            # the index of the centre point is [0, 0]. For example, a
            # 5x5 mask would start at [-2, -2], and go up to [2, 2]
            for m, maskRow in enumerate(mask, start=(maskVerticalCentre - maskRows)):
                for n, maskValue in enumerate(maskRow, start=(maskHorizontalCentre - maskColumns)):

                    # add to the output array the mask value multiplied
                    # by the pixel value underneath it
                    outputArray[i, j] += inputArray[i + m, j + n] * maskValue

    return outputArray

# fast convolution for testing purposes
def fastConvolveWithMask(inputArray, mask):
    from scipy import signal
    return signal.convolve2d(inputArray, mask)

def medianFilter(inputArray, maskSize):

    # check that maskSize is odd
    if maskSize % 2 == 0:
        raise ValueError("mask size is not an odd number")

    # check inputArray is valid
    if inputArray is None:
        raise ValueError("input array is not valid")

    # get centre of mask (zero-indexed, so a 5x5 mask would have a centre 2)
    maskCentre = math.ceil((maskSize-1)/2)

    # create empty output array with same shape as input array
    outputArray = np.zeros(inputArray.shape)

    # -- populate the output array --
    # loop over rows, ignoring half the mask at the top and bottom
    # tqdm adds a progress bar
    for i, row in enumerate(tqdm(inputArray[maskCentre:-maskCentre]),
                            start=maskCentre):

        # loop over the pixels in the row, ingoring half the mask
        # at each end
        for j, _pixel in enumerate(row[maskCentre:-maskCentre],
                                   start=maskCentre):

            # create empty mask
            mask = np.zeros((maskSize,maskSize))

            # -- fill mask --
            # loop over mask in 2d, with [0,0] at the centre. For
            # a 5x5 mask, this would loop from [-2,-2] to [2,2]
            for m in range(-maskCentre,maskSize-maskCentre):
                for n in range(-maskCentre,maskSize-maskCentre):

                    # get the pixel value under each element of the mask
                    mask[m+maskCentre,n+maskCentre] = inputArray[i+m,j+n]

            # calculate the median and put it in the output array
            outputArray[i,j] = np.median(mask)

    return outputArray

def crimmins(imageArray, iterations):

    # vectors representing the four directions:
    # E-W, NW-SE, N-S, NE-SW. Each vector is in
    # the format [E,S], so [1,1] is one step
    # East and one step South
    directions = [[1,0], [1,1], [0,1], [-1,1]]

    # create progress bar object
    pbar = tqdm(total=100, bar_format="{percentage:.1f}%|{bar}| elapsed: {elapsed} | remaining: {remaining}")
    # get size of progess bar steps as a percentage
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

                    # get value of previous pixel in this direction by subtracting the East 
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
    return imageArray