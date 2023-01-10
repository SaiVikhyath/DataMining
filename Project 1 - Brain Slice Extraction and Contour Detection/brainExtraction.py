# Author: Sai Vikhyath K
# Date: 25 September, 2022


from PIL import Image, ImageFilter
from cmath import inf
import numpy as np
import shutil
import math
import sys
import cv2
import os


"""
Design and Implementation:
-> Multi-Template matching is used to detect all the R's.
-> Co-ordinates of the R's are obtained from template matching.
-> Width and Heigth between adjacent R's is extracted from the co-ordinates.
-> Each brain slice is of size (width*height).
-> This width and height are used to extract brain slices from the images.
-> Once th brain slice is extracted, contour plotting is performed on the sliced image.
"""


imagesPath = r"Data\\"


def deleteExistingDirectories():
    """ Delete Slices and Boundaries directories if they already exist. """
    try:
        if os.path.exists(r"Slices\\"):
            shutil.rmtree(r"Slices\\")
        if os.path.exists(r"Boundaries\\"):
            shutil.rmtree(r"Boundaries\\")
    except:
        print("Unable to delete existing directories")


def getImagesList(imagesPath):
    """ Get the list of all IC thresh images.
        For each image in the directory, check if the image starts with "IC_" and ends with "_thresh.png".
    """
    imagesList = []
    for image in os.listdir(imagesPath):
        if image.startswith("IC_") and image.endswith("_thresh.png"):
            imagesList.append(image)
    return imagesList
    

def checkBlackImage(image):
    """ Count the total number of pixels in the image. Count the number pixels that are black. i.e. RGB is (0, 0, 0) => sum(RGB) = 0.
        If the total number of pixels is equal to the number of black pixels => Image is black => Image has no brain slice.
    """
    height, width, channels = image.shape
    if int(height) * int(width) * int(channels) == np.sum(image == 0):
        return True
    else:
        return False


def createOutline(slicedImage, imageNo, slicedImageCount):
    """ Generate the contour of the brain for each sliced image """
    grayImage = cv2.cvtColor(slicedImage, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY)
    contour = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(slicedImage, contour[0], -1, (0, 0, 255), 1)
    cv2.imwrite("Boundaries\\" + "IC_" + str(imageNo + 1) + "//" + str(slicedImageCount) + ".png", slicedImage)


def extractSlicesAndBoundaries(imagesPath, imagesList):
    """ Extract brain slices from each image. Then a contour is drawn for each of the extracted brain slice image.
        Template matching is used to extract the co-ordinates of all the R's in the image.
        Width and Height between the R's is extracted.
        This width and height is used iteratively to extract the brain slices which are stored in Slices directory.
        For each of the extracted brain slice, a contour is drawn using createOutline function.
    """
    template = cv2.imread("template.png", 0)
    imageNo = 0
    for imgName in imagesList:
        image = cv2.imread(imagesPath + imgName)
        height, width, _ = image.shape
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(grayImage, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        rLocations = zip(*loc[::-1])
        firstR = list(rLocations)[0]
        firstRw = firstR[0]
        firstRh = firstR[1]
        cropHeight = firstRh
        cropWidth = 0    
        brainHeight = inf
        brainWidth = inf
        for pt in zip(*loc[::-1]):
            if firstRw == pt[0] and firstRh == pt[1]:
                continue
            elif firstRw == pt[0]:
                brainHeight = min(brainHeight, pt[1] - firstRh)
            elif firstRh == pt[1]:
                cropWidth = max(cropWidth, pt[0])
                brainWidth = min(brainWidth, pt[0] - firstRw)
        image = image[cropHeight:height, 0:cropWidth+brainWidth]
        croppedHeight, croppedWidth, _ = image.shape
        if not os.path.exists(""r"Slices\\"):
            os.mkdir("Slices\\")
        if not os.path.exists(r"Slices\\" + "IC_" + str(imageNo + 1)):
            os.mkdir("Slices\\" + "IC_" + str(imageNo + 1) + "\\")
        if not os.path.exists(""r"Boundaries\\"):
            os.mkdir("Boundaries\\")
        if not os.path.exists(r"Boundaries\\" + "IC_" + str(imageNo + 1)):
            os.mkdir("Boundaries\\" + "IC_" + str(imageNo + 1) + "\\")
        cumulativeHeight = 0
        slicedImageCount = 0
        while cumulativeHeight + brainHeight < croppedHeight:
            cumulativeWidth = 0
            while cumulativeWidth + brainWidth < croppedWidth:
                slicedImage = image[cumulativeHeight + 8:cumulativeHeight + brainHeight, cumulativeWidth:cumulativeWidth + brainWidth]
                isBlackImage = checkBlackImage(slicedImage)
                if not isBlackImage:
                    slicedImageCount += 1
                    cv2.imwrite("Slices\\" + "IC_" + str(imageNo + 1) + "\\" + str(slicedImageCount) + ".png", slicedImage)
                    createOutline(slicedImage, imageNo, slicedImageCount)
                cumulativeWidth += brainWidth
            cumulativeHeight += brainHeight
        imageNo += 1


if __name__ == "__main__":
    deleteExistingDirectories()
    imagesList = getImagesList(imagesPath)
    extractSlicesAndBoundaries(imagesPath, imagesList)