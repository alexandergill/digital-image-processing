import math
import numpy as np
from tqdm import tqdm

def convoleWithMask(inputArray, mask):

    # check that mask is one- or two-dimensional
    if len(mask.shape) > 2:
        raise ValueError("mask has more than two dimensions")

    # check that mask is odd-numbered in both dimensions
    for dimension in mask.shape:
        if (dimension % 2) == 0:
            raise ValueError("mask is not odd in both dimensions")

    # get mask dimensions
    [maskRowLength, maskColLength] = mask.shape
    maskRowCentre = math.ceil(maskRowLength/2)
    maskColCentre = math.ceil(maskColLength/2)

    # create empty output array of same shape as input
    outputArray = np.zeros(inputArray.shape)

    for i, row in enumerate(tqdm(inputArray[maskRowCentre:-maskRowCentre]), start=maskRowCentre):
        for j, column in enumerate(row[maskColCentre:-maskColCentre], start=maskColCentre):
            for m, maskRow in enumerate(mask, start=(maskRowCentre - maskRowLength)):
                for n, maskValue in enumerate(maskRow, start=(maskColCentre - maskColLength)):
                    outputArray[i, j] += inputArray[i + m, j + n] * maskValue

    return outputArray
