import numpy as np

from Utilities import Utilities
from PreProcessing import PreProcessing
import DirectorySetup

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

class Model():
    def __init__(self):
        self.model = self.buildModel()

        self.preProcessing = PreProcessing()
        self.trainingKeyPoints, self.testingKeyPoints, self.trainingLabels, self.testingLabels= self.preProcessing.split_training_testing_set()



    def buildModel(self):
        utilities = Utilities()
        signListLength = len(utilities.getListOfSigns())

        # print(len(signList))

        model = Sequential()
        model.add(LSTM(64, return_sequences = True, activation ='relu', input_shape = (30,1662)))
        model.add(LSTM(128, return_sequences = True, activation ='relu'))
        model.add(LSTM(64, return_sequences = False, activation ='relu'))
        model.add(Dense(64,activation= 'relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(signListLength, activation='softmax'))

        model.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])

        return model

    def trainModel(self):
        #set up log Directory for seeing what is happening inside the model
        logDirectory = DirectorySetup.LogDirectory
        tbCallback = TensorBoard(log_dir=logDirectory)



        self.model.fit(self.trainingKeyPoints, self.trainingLabels, epochs = 250, callbacks=[tbCallback])

    def getModelSummary(self):
        summary = self.model.summary()
        return summary

    def testModel(self):
        return self.model.predict(self.testingKeyPoints)

    def getLabelMapping(self):
        utilities = Utilities()
        signList = utilities.getListOfSigns()
        labelMap = {}
        for number, sign in enumerate(signList):
            signParts = sign.split('_')
            labelMap[signParts[1]] = [number, signParts[0]]

        return labelMap

if __name__ == '__main__':
    model = Model()

    # print(model.X_test.shape)
    # print(model.Y_test.shape)
    # print(model.X_train.shape)
    # print(model.Y_test.shape)

    # labelMap = model.getLabelMapping()
    # print(labelMap)

    model.trainModel()
    summary = model.getModelSummary()
    print(summary)

    # validaionLoss, validationAccuracy = model.model.evaluate(model.validationKeyPoints, model.validationLabels)

    testLoss, testAccuray = model.model.evaluate(model.testingKeyPoints, model.testingLabels)
    print(testAccuray)








