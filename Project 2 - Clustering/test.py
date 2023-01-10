# Author: Sai Vikhyath K
# Date: 24 October, 2022


"""
Design and Implementation:
-> Multi-Template matching is used to detect all the R's.
-> Co-ordinates of the R's are obtained from template matching.
-> Width and Heigth between adjacent R's is extracted from the co-ordinates.
-> Each brain slice is of size (width*height).
-> This width and height are used to extract brain slices from the images.
-> Remove gray pixels from the extracted slices. (For a gray pixel, the RGB components of the pixel are equal)
-> Perform DBSCAN on the image after removal of gray pixels.
"""


import clustering as clus
from cmath import inf
import sys


imagesPath = r"testPatient\\"


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    clus.deleteExistingDirectories()
    imagesList = clus.getImagesList(imagesPath)
    clus.extractSlices(imagesPath, imagesList)