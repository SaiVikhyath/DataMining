# Author: Sai Vikhyath K
# Date: 21 November 2022


"""
Description:
Images of resting state networks and noise are provided along with the labels.
Need to build a convolutional neural network to classify the images into appropriate category.
Labels have to preprocessed to convert the values greater than 1 to 1. Since, Binary classification is the motive.
Images are first resized into the size (256 x 256) which are then fed into CNN.
"""



from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AvgPool2D, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np
import glob
import sys
import cv2
import os
import re



class Classification:
    """
    Preprocesses the labels of the data.
    Builds a Convolutional Neural Network.
    """


    def __init__(self, imagesPath) -> None:
        """ Read input of all the images into numpy array.
            Each image is resized into size (256 x 256 x 3)
            These images input are stored in a NumPy array of shape (number_of_images x 256 x 256 x 3).
            Pixel values of the images are normalized by dividing each value with 255.
        """

        trainData = []
        
        os.chdir(imagesPath)

        for directory in sorted(glob.glob('*'), key=numericalSort):
            if directory.startswith("Patient_") and (not directory.endswith("_Labels.csv")):
                path = directory
            else:
                continue
            
            os.chdir(path)
            
            for image in sorted(glob.glob('*'), key=numericalSort):
                if image.startswith("IC_") and image.endswith("_thresh.png"):
                    number = image.split("_")[1]
                    # print(image, number, path)
                    imageData = cv2.imread(image)
                    imageData = cv2.resize(imageData, (256, 256))
                    trainData.append(imageData)

            os.chdir("..")
            
        self.trainData = np.array(trainData, dtype="float32")
        self.trainData /= 255

        # print(self.trainData.shape)


    def changeLabels(self, path) -> None:
        """ Read labels of the images as input.
            Change the values of the labels to 1 wherever the label is greater than 1 to facilitate binary classification.
            The labels are stored in a NumPy array.
        """

        labelsDF = pd.DataFrame(columns=["IC", "Label"])
        
        for fileName in sorted(glob.glob('*'), key=numericalSort):
            
            if fileName.startswith("Patient_") and fileName.endswith("_Labels.csv"):
                file = pd.read_csv(fileName)
                # print(fileName)
                file["Label"] = file["Label"].replace(to_replace=[1, 2, 3], value=1)
                # print(any(file["Label"] == 2))
                # print(any(file["Label"] == 3))
                labelsDF = pd.concat([labelsDF, file])
        
        self.labels = np.array(labelsDF["Label"], dtype="float32")
        
        # print(self.labels.shape)


    def CNN(self) -> None:
        """ Build a Convolutional Neural Network.
            There are 4 Convolution2D layers. With first two layers having 32 filters and the filter size being 2 x 2 and activation functions being Rectified Linear Unit.
            Between each pair of Convolution2D layer, Average Pooling is applied with a window of size 2 x 2.
            Then the nodes are flattened.
            Then exists a dense layer of 512 nodes which are fully connected to the previous layer and the activation function being Rectified Linear Unit.
            The output layer consists of a solitary node with sigmoid activation function.
            Then the model is compiled with Adams optimizer, learning_rate being 0.001, loss being binary_crossentropy and metric used is accuracy.
            Then the model is trained for 50 epochs with batch_size of 12 and saved.
            Train and test accuracy is computed along with precision, sensitivity and specificity.
        """

        trainData = self.trainData[:400]
        testData = self.trainData[400:]
        trainLabels = self.labels[:400].reshape((len(self.labels[:400]), 1))
        testLabels = self.labels[400:].reshape((len(self.labels[400:]), 1))

        print("Train data size: ", trainData.shape)
        print("Test data size: ", testData.shape)
        print("Train labels size: ", trainLabels.shape)
        print("Test labels size: ", testLabels.shape)


        model = Sequential([
            Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=(256, 256, 3)),
            AvgPool2D((2, 2)),
            Conv2D(filters=32, kernel_size=(2, 2), activation='relu'),
            AvgPool2D((2, 2)),
            Conv2D(filters=64, kernel_size=(2, 2), activation='relu'),
            AvgPool2D((2, 2)),
            Conv2D(filters=64, kernel_size=(2, 2), activation='relu'),
            AvgPool2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics = ['acc'])

        model.fit(trainData, trainLabels, epochs=50, batch_size=12, verbose=1, validation_data=(testData, testLabels))
        
        model.save("modelCNN2")

        # model = load_model('modelCNN')

        trainScore = model.evaluate(trainData, trainLabels, verbose=1)

        print('Train loss:', trainScore[0])
        print('Train accuracy:', trainScore[1])

        testScore = model.evaluate(testData, testLabels, verbose=0)

        print('Test loss:', testScore[0])
        print('Test accuracy:', testScore[1])
        print(np.rint(model.predict(testData)))
        
        tn, fp, fn, tp = confusion_matrix(testLabels, np.rint(model.predict(testData))).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp + fn)

        print("Precision: ", precision_score(testLabels, np.rint(model.predict(testData))))
        print("Specificity: ", specificity)
        print("Sensitivity: ", sensitivity)

        print(model.summary())


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    imagesPath = r"PatientData-2\PatientData"
    classifier = Classification(imagesPath)
    labelsPath = r"PatientData-2\PatientData"
    classifier.changeLabels(labelsPath)
    classifier.CNN()