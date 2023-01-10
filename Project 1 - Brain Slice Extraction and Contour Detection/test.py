# Author: Sai Vikhyath K
# Date: 28 September, 2022


"""
Design and Implementation:
-> Multi-Template matching is used to detect all the R's.
-> Co-ordinates of the R's are obtained from template matching.
-> Width and Heigth between adjacent R's is extracted from the co-ordinates.
-> Each brain slice is of size (width*height).
-> This width and height are used to extract brain slices from the images.
-> Once th brain slice is extracted, contour plotting is performed on the sliced image.
"""


from PIL import Image, ImageFilter
import brainExtraction as be
from cmath import inf
import numpy as np
import shutil
import math
import sys
import cv2
import os


imagesPath = r"testPatient\\"


if __name__ == "__main__":
    be.deleteExistingDirectories()
    imagesList = be.getImagesList(imagesPath)
    be.extractSlicesAndBoundaries(imagesPath, imagesList)