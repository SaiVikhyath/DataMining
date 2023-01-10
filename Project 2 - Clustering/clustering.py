# Author: Sai Vikhyath K
# Date: 24 October, 2022

from collections import defaultdict
from tkinter import Image
from typing import List
from cmath import inf
import numpy as np
import shutil
import math
import cv2
import csv
import sys
import os


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


imagesPath = r"Data\\"


def deleteExistingDirectories() -> None:
    """ Delete Slices and Clusters directories if they already exist. """
    try:
        if os.path.exists(r"Slices\\"):
            shutil.rmtree(r"Slices\\")
        if os.path.exists(r"Clusters\\"):
            shutil.rmtree(r"Clusters\\")
    except:
        print("Unable to delete existing directories")
        quit()


def getImagesList(imagesPath: str) -> List:
    """ Get the list of all IC thresh images.
        For each image in the directory, check if the image starts with "IC_" and ends with "_thresh.png".
    """
    imagesList = []
    for image in os.listdir(imagesPath):
        if image.startswith("IC_") and image.endswith("_thresh.png"):
            imagesList.append(image)
    return imagesList
    

def checkBlackImage(image: Image) -> bool:
    """ Count the total number of pixels in the image. Count the number pixels that are black. i.e. RGB is (0, 0, 0) => sum(RGB) = 0.
        If the total number of pixels is equal to the number of black pixels => Image is black => Image has no brain slice.
    """
    height, width, channels = image.shape
    if int(height) * int(width) * int(channels) == np.sum(image == 0):
        return True
    else:
        return False


def euclideanDistance(p1: tuple, p2: tuple) -> float:
    """ Compute Euclidean Distance between two points
        Euclidean Distance = sqrt((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
    """
    return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))


def retrieveNeighbours(indexP1: int, points: List, epsilon: float) -> List:
    """ Get all the neighbours of point1 within the epsilon distance"""
    neighbourIndices = []
    for indexP2, p2 in enumerate(points):
        if indexP1 != indexP2 and euclideanDistance(points[indexP1], points[indexP2]) <= epsilon:
            neighbourIndices.append(indexP2)
    return neighbourIndices



def visitNeighbours(indexP1: int, points: List, clusterIndex: int, clusterLabels: List, minimumPoints: int, epsilon: float) -> None:
    """ Visit all neighbours of point1 and assign the cluster index"""
    for indexP2 in retrieveNeighbours(indexP1, points, epsilon):
        if clusterLabels[indexP2] == -1:
            clusterLabels[indexP2] = clusterIndex
            if len(retrieveNeighbours(indexP2, points, epsilon)) >= minimumPoints:
                visitNeighbours(indexP2, points, clusterIndex, clusterLabels, minimumPoints, epsilon)


def dbscan(slicedImage: Image, imageNo: int, slicedImageCount: int, clustersInIC: dict) -> dict:
    """Perform DBSCAN on the image"""
    epsilon = 2
    minimumPoints = 5
    points = []
    height, width, _ = slicedImage.shape

    for row in range(height):
        for column in range(width):
            if slicedImage[row][column][0] != slicedImage[row][column][1] != slicedImage[row][column][2]:
                points.append((row, column))

    coloredImg = np.zeros((height, width, 3))
    pointIndex = 0

    for i in range(height):
        for j in range(width):
            if (i, j) in points:
                coloredImg[i][j] = np.array([0, 255, 0])
                pointIndex += 1
            else:
                coloredImg[i][j] = np.array([0, 0, 0])

    cv2.imwrite("Clusters\\" + "IC_" + str(imageNo + 1) + "//" + str(slicedImageCount) + ".png", coloredImg)

    clusterIndex = 0
    clusterLabels = [-1] * len(points)
    
    for indexP1, p1 in enumerate(points):
        if clusterLabels[indexP1] != -1:
            continue
        if len(retrieveNeighbours(indexP1, points, epsilon)) <= minimumPoints:
            clusterLabels[indexP1] = clusterIndex
            visitNeighbours(indexP1, points, clusterIndex, clusterLabels, minimumPoints, epsilon)
        clusterIndex += 1

    clusters = defaultdict(list)
    
    for i in range(len(clusterLabels)):
        clusters[clusterLabels[i]].append(points[i])
    
    bigClusters = defaultdict(list)
    
    for k, v in clusters.items():
        if len(v) >= 135:
            bigClusters[k].append(v)
    
    clustersInIC["IC_" + str(imageNo + 1)].append(tuple([slicedImageCount, len(bigClusters)]))
    
    return clustersInIC


def createCSV(clustersInIC: dict) -> None:
    """ Create a csv for each IC with number of clusters against each image"""
    for k, v in clustersInIC.items():
        f = open("Clusters\\" + str(k) + "\\" + str(k) + ".csv", "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["SliceNumber", "ClusterCount"])
        for i, c in enumerate(v):
            writer.writerow([str(c[0]), str(c[1])])
        f.close()


def extractSlices(imagesPath: str, imagesList: List) -> None:
    """ Extract brain slices from each image. Then a contour is drawn for each of the extracted brain slice image.
        Template matching is used to extract the co-ordinates of all the R's in the image.
        Width and Height between the R's is extracted.
        This width and height is used iteratively to extract the brain slices which are stored in Slices directory.
        For each of the extracted brain slice, a contour is drawn using createOutline function.
    """
    template = cv2.imread("template.png", 0)
    imageNo = 0
    clustersInIC = defaultdict(list)
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
        if not os.path.exists(""r"Clusters\\"):
            os.mkdir("Clusters\\")
        if not os.path.exists(r"Clusters\\" + "IC_" + str(imageNo + 1)):
            os.mkdir("Clusters\\" + "IC_" + str(imageNo + 1) + "\\")
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
                    clustersInIC = dbscan(slicedImage, imageNo, slicedImageCount, clustersInIC)
                    createCSV(clustersInIC)
                cumulativeWidth += brainWidth
            cumulativeHeight += brainHeight
        imageNo += 1
    

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    deleteExistingDirectories()
    imagesList = getImagesList(imagesPath)
    extractSlices(imagesPath, imagesList)