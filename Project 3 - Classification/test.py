# Author: Sai Vikhyath K
# Date: 21 November 2022


"""
Description:
Images of resting state networks and noise are provided along with the labels.
Need to build a convolutional neural network to classify the images into appropriate category.
Labels have to preprocessed to convert the values greater than 1 to 1. Since, Binary classification is the motive.
Images are first resized into the size (256 x 256) which are then fed into CNN.
"""


from keras.models import load_model
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import glob
import csv
import sys
import cv2
import os
import re



numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def readImages(imagesPath) -> None:
    """ Read input images from test_Data folder into a NumPy array.
        Each pixel is normalized by dividing by 255.
    """
    
    testData = []
    
    os.chdir(imagesPath)

    for image in sorted(glob.glob('*'), key=numericalSort):
        if image.startswith("IC_") and image.endswith("_thresh.png"):
            number = image.split("_")[1]
            imageData = cv2.imread(image)
            imageData = cv2.resize(imageData, (256, 256))
            testData.append(imageData)

    os.chdir("../..")

    testData = np.array(testData, dtype="float32")
    testData /= 255

    return testData


def readLabels(labelsPath):
    """ Labels are read from the test_Labels.csv.
        All labels with value greater than 1 are converted to 1.
    """

    os.chdir(labelsPath)

    labels = []

    labelsDF = pd.DataFrame(columns=["IC", "Label"])

    for fileName in sorted(glob.glob('*'), key=numericalSort):
            
        if fileName.startswith("test_") and fileName.endswith("_Labels.csv"):
            file = pd.read_csv(fileName)
            # print(fileName)
            file["Label"] = file["Label"].replace(to_replace=[1, 2, 3], value=1)
            # print(any(file["Label"] == 2))
            # print(any(file["Label"] == 3))
            labelsDF = pd.concat([labelsDF, file])
    
    labels = np.array(labelsDF["Label"], dtype="float32")

    os.chdir("..")

    return labels


def prediction(testImages, testLabels):
    """ Load the trained model.
        Predict the test data.
        Compute confusion matrix.
        Using confusion matrix, compute accuracy, precision, sensitivity, specificity.
        Store the metrics and predicted output into csv files.
    """


    model = load_model('modelCNN2')

    predictions = np.rint(model.predict(testImages))

    tn, fp, fn, tp = confusion_matrix(testLabels, np.rint(model.predict(testImages))).ravel()

    accuracy = (tp + tn) / (tn + fp + fn + tp)
    precision = (tp) / (tp + fp)
    sensitivity = (tp) / (tp + fn)
    specificity = (tn) / (tn + fp)

    print("Accuracy : ", accuracy)
    print("Precision : ", precision)
    print("Sensitivity : ", sensitivity)
    print("Specificity : ", specificity)
    
    if os.path.exists("Metrics.csv"):
        os.remove("Metrics.csv")

    try:
        f = open("Metrics.csv", "w", newline="")
        writer = csv.writer(f)
    except:
        print("Unable to create Metrics.csv.")
        quit()

    writer.writerow(["Metric", "Value"])
    writer.writerow(["Accuracy", str(accuracy * 100) + " %"])
    writer.writerow(["Precision", str(precision * 100) + " %"])
    writer.writerow(["Sensitivity", str(sensitivity * 100) + " %"])
    writer.writerow(["Specificity", str(specificity * 100) + " %"])

    if os.path.exists("Results.csv"):
        os.remove("Results.csv")

    try:
        f = open("Results.csv", "w", newline="")
        writer = csv.writer(f)
    except:
        print("Unable to create Results.csv.")
        quit()

    writer.writerow(["IC_Number", "Label"])
    for idx, pred in enumerate(predictions):
        writer.writerow([str(idx + 1), str(pred)[1]])
        


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    imagesPath = r"testPatient\test_Data"
    testImages = readImages(imagesPath)
    labelsPath = r"testPatient"
    testLabels = readLabels(labelsPath)
    testLabels = testLabels.reshape((len(testLabels), 1))
    prediction(testImages, testLabels)