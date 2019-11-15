import math
import numpy as np
from tqdm import tqdm

def convoleWithMask(inputArray, mask):

    print(mask)

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

    # loop over the rows in the image, ignoring half the mask at the
    # top and the bottom. 'tqdm' adds a progress bar. 'start' gives
    # the correct index within the loop.
    for i, row in enumerate(tqdm(inputArray[maskVerticalCentre:-maskVerticalCentre]), start=maskVerticalCentre):

        # loop over the columns in the same way
        for j, column in enumerate(row[maskHorizontalCentre:-maskHorizontalCentre], start=maskHorizontalCentre):

            # loop over the rows of the mask. 'start' is set so that
            # the index of the centre point is 0, 0. For example, a
            # 5x5 mask would start at -2, and go up to 2
            for m, maskRow in enumerate(mask, start=(maskVerticalCentre - maskRows)):
                for n, maskValue in enumerate(maskRow, start=(maskHorizontalCentre - maskColumns)):

                    # add to the output array the mask value multiplied
                    # by the pixel value underneath it
                    outputArray[i, j] += inputArray[i + m, j + n] * maskValue

    return outputArray
