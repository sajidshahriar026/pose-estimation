from Utilities import Utilities
import DirectorySetup
import os
import numpy as np




class FetchKeyPointsAndLabels():
    def __init__(self):
        self.utilities = Utilities()
        self.signList = self.utilities.getListOfSigns()
        self.VideoStoreDirectory = DirectorySetup.VideoStoreDirectory
        self.KeyPointsStoreDirectory = DirectorySetup.KeyPointsStoreDirectory
        self.ImageStoreDirectory = DirectorySetup.ImageStoreDirectory
        self.SignListDirectory = DirectorySetup.SignListDirectory
        self.labelMap = self.getLabelMapping()

    def getLabelMapping(self):
        labelMap = {}
        for number, sign in enumerate(self.signList):
            signParts = sign.split('_')
            labelMap[signParts[1]] = [number, signParts[0]]

        # print(labelMap)
        return labelMap

    def returnSortedKeyPoints(self, subDirectoryPath):
        keyPointSequenceList = []
        for keyPoint in os.listdir(subDirectoryPath):
            keyPointNamePart = keyPoint.split('_')
            keyPointSequenceNumber = keyPointNamePart[2].split('.')[0]
            keyPointSequence = (int(keyPointSequenceNumber), keyPoint)
            keyPointSequenceList.append(keyPointSequence)

        keyPointSequenceList.sort()
        # print(keyPointSequenceList)
        return keyPointSequenceList

    def getKeyPointsAndLabels(self):
        keyPointsdirectory = self.KeyPointsStoreDirectory   #./keypoints
        keyPointListForAllVideos = []
        labels = []

        for directory in os.listdir(keyPointsdirectory):
            directoryName = directory
            # print(directoryName)
            if directory.startswith('videoCapture'):  # ./keypoints/videoCap
                continue
            else:

                directoryPath = os.path.join(keyPointsdirectory, directory)
                # print(directoryPath)

                for subDirectory in os.listdir(directoryPath):
                    subDirectoryPath = os.path.join(directoryPath, subDirectory)

                    # print(subDirectoryPath)

                    singleVideoKeyPointList = []
                    sortedKeyPointSequenceList = self.returnSortedKeyPoints(subDirectoryPath)

                    for sequence, keypoint in sortedKeyPointSequenceList:
                        keyPointPath = os.path.join(subDirectoryPath, keypoint)
                        keyPointArray = np.load(keyPointPath)
                        singleVideoKeyPointList.append(keyPointArray)

                    keyPointListForAllVideos.append(singleVideoKeyPointList)
                    labels.append(self.labelMap[directoryName][0])

        # print(np.array(keyPointListForAllVideos).shape)
        # print(labels)
        # print(np.array(labels).shape)

        return (keyPointListForAllVideos,labels)


if __name__ == '__main__':
    fetchKeyPointsAndLabels = FetchKeyPointsAndLabels()
    # print(fetchKeyPointsAndLabels.labelMap)
    keyPointsForAllVideos, labels = fetchKeyPointsAndLabels.getKeyPointsAndLabels()

    # print(keyPointsForAllVideos)
    print(labels)
    print(np.array(keyPointsForAllVideos).shape)
    print(np.array(labels).shape)
    print(fetchKeyPointsAndLabels.labelMap)
