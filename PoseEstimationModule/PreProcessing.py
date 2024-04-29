from FetchKeyPointsAndLabels import FetchKeyPointsAndLabels
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


class PreProcessing():
    def __init__(self):
        fetchKeyPointsAndLabels = FetchKeyPointsAndLabels()

        self.keyPoints, self.labels = fetchKeyPointsAndLabels.getKeyPointsAndLabels()
        self.keyPoints = np.array(self.keyPoints)
        # print(self.labels)
        self.labels = to_categorical(self.labels).astype(int)
        # print(self.labels)

    def split_training_testing_set(self):
        trainingKeyPoints, testingKeyPoints, trainingLabels, testingLabels  = train_test_split(self.keyPoints, self.labels, test_size=0.20)
        return (trainingKeyPoints, testingKeyPoints, trainingLabels, testingLabels)

if __name__ == "__main__":

    preProcessing = PreProcessing()
    trainingKeyPoints, testingKeyPoints, trainingLabels, testingLabels = preProcessing.split_training_testing_set()


    print(trainingKeyPoints.shape)
    print(testingKeyPoints.shape)

    print(trainingLabels.shape)
    print(testingLabels.shape)