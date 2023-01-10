# Author: Sai Vikhyath K
# Date: 25 September, 2022

from PIL import Image, ImageFilter
import numpy as np
import shutil
import math
import cv2
import os


"""
-> Since the images are of different sizes, resize all the images to have the same aspect ratio. i.e 1024 * 851.
-> This aspect ratio of 1024 * 851 has been decided based on mean of all the aspect ratio's of the images.
-> Save all the resized images in the folder Resized Images.
-> Crop the resized images to remove the top and right regions of the image that do not have brain slices.
-> These cropped images are equally split by the no of rows and columns of images to extract slices.
-> For each sliced image, an outline for the brain boundary is drawn.
"""

imagesPath = r"PatientData-1\\PatientData\Data\\"
resizedImagesPath = r"PatientData-1\\PatientData\\Resized Images\\"
croppedImagesPath = r"PatientData-1\\PatientData\\Cropped Images\\"
slicedImagesPath = r"PatientData-1\\PatientData\\Slices\\"
totalImages = 112


def deleteExistingDirectories():
    """ Remove previous or already existing directory structures """
    try:
        if os.path.exists(r"PatientData-1\\PatientData\\Resized Images\\"):
            shutil.rmtree(r"PatientData-1\\PatientData\\Resized Images\\")
        if os.path.exists(r"PatientData-1\\PatientData\\Cropped Images\\"):
            shutil.rmtree(r"PatientData-1\\PatientData\\Cropped Images\\")
        if os.path.exists(r"PatientData-1\\PatientData\\Slices\\"):
            shutil.rmtree(r"PatientData-1\\PatientData\\Slices\\")
    except:
        print("Unable to delete existing directories")


def createDirectoryStructures():
    """ Create directories for resized and cropped images """
    try:
        if not os.path.exists(r"PatientData-1\\PatientData\\Resized Images\\"):
            os.mkdir(r"PatientData-1\\PatientData\\Resized Images\\")
        if not os.path.exists(r"PatientData-1\\PatientData\\Cropped Images\\"):
            os.mkdir(r"PatientData-1\\PatientData\\Cropped Images\\")
    except:
        print("Unable to create the directories: Resized Images and Cropped Images")


def checkBlackImage(croppedImage):
    """ Count the total number of pixels in the image. Count the number pixels that are black. i.e. RGB is (0, 0, 0) => sum(RGB) = 0
        If the total number of pixels is almost same as the number of black pixels => Image has no brain slice.
    """
    h, w, channels = croppedImage.shape
    if int(h) * int(w) * int(channels) == np.sum(croppedImage == 0):
        return True
    else:
        return False


def createOutline(slicedImg, imageNo, slicedImageCount):
    """ Generate the contour of the brain for each sliced image """
    gray = cv2.cvtColor(slicedImg, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cont = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(slicedImg, cont[0], -1, (0, 0, 255), 2)
    cv2.imwrite("PatientData-1\\PatientData\\Boundaries\\" + "IC_" + str(imageNo + 1) + "_thresh\\" + "sliced_" + str(imageNo + 1) + "_image_ " + str(slicedImageCount) + ".png", slicedImg)


def extractSlicesAndBoundaries():
    """ Resize all images to a common standard aspect ratio
        Crop the resized images to remove the top and right portions of the image which do not have brain slices
        Extract slices of brain from the cropped images
        Draw contours for the brains for each sliced image.
    """
    for imageNo in range(totalImages):
        img = cv2.imread(imagesPath + "IC_" + str(imageNo + 1) + "_thresh.png")
        # height, width, _ = img.shape
        resizedImg = cv2.resize(img, (1024, 854))
        height, width, _ = resizedImg.shape
        cv2.imwrite(resizedImagesPath + "IC_" + str(imageNo + 1) + "_thresh.png", resizedImg)
        heightToBeCut = math.ceil((1/7) * height) + 18  # Image is split into 7 equal horizontal bars. Top bar does not have brain images.
        widthToBeCut = 72   # To ignore the right side vertical bar.
        croppedImg = resizedImg[heightToBeCut:height, 0:width-widthToBeCut]
        cv2.imwrite(croppedImagesPath + "IC_" + str(imageNo + 1) + "_thresh.png", croppedImg)
        croppedImgHeight, croppedImgWidth, _ = croppedImg.shape
        eachImgWidth = int(croppedImgWidth / 8)
        eachImgHeight = int(croppedImgHeight / 6)
        if not os.path.exists(r"PatientData-1\\PatientData\\Slices\\"):
            os.mkdir("PatientData-1\\PatientData\\Slices\\")
        if not os.path.exists(r"PatientData-1\\PatientData\\Slices\\" + "IC_" + str(imageNo + 1) + "_thresh"):
            os.mkdir("PatientData-1\\PatientData\\Slices\\" + "IC_" + str(imageNo + 1) + "_thresh\\")
        if not os.path.exists(r"PatientData-1\\PatientData\\Boundaries\\"):
            os.mkdir("PatientData-1\\PatientData\\Boundaries\\")
        if not os.path.exists(r"PatientData-1\\PatientData\\Boundaries\\" + "IC_" + str(imageNo + 1) + "_thresh"):
            os.mkdir("PatientData-1\\PatientData\\Boundaries\\" + "IC_" + str(imageNo + 1) + "_thresh\\")
        slicedImageCount = 0
        for i in range(0, croppedImgHeight, eachImgHeight):
            for j in range(0, croppedImgWidth, eachImgWidth):
                slicedImg = croppedImg[i:i+eachImgHeight, j+5:j+eachImgWidth-10]
                isBlackImage = checkBlackImage(slicedImg)
                if not isBlackImage:
                    slicedImageCount += 1
                    cv2.imwrite("PatientData-1\\PatientData\\Slices\\" + "IC_" + str(imageNo + 1) + "_thresh\\" + "sliced_" + str(imageNo + 1) + "_image_ " + str(slicedImageCount) + ".png", slicedImg)
                    createOutline(slicedImg, imageNo, slicedImageCount)


if __name__ == "__main__":
    deleteExistingDirectories()
    createDirectoryStructures()
    extractSlicesAndBoundaries()