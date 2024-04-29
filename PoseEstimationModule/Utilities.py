import os
import DirectorySetup

class Utilities:
    def __init__(self):
        self.VideoStoreDirectory = DirectorySetup.VideoStoreDirectory
        self.KeyPointsStoreDirectory = DirectorySetup.KeyPointsStoreDirectory
        self.ImageStoreDirectory = DirectorySetup.ImageStoreDirectory
        self.SignListDirectory = DirectorySetup.SignListDirectory

    def getListOfSigns(self):
        signListFilePath = os.path.join(self.SignListDirectory, 'signList.txt')
        if os.path.exists(signListFilePath):
            signListFile = open(signListFilePath, 'r', encoding='UTF-8')
            signList = []
            for line in signListFile:
                if line[-1] == '\n':
                    line = line[:-1]
                signList.append(line)

        signListFile.close()
        return signList

    def getVideoList(self):
        signList = self.getListOfSigns()
        videoList = []
        # iterate over the sign list
        for sign in signList:
            sign = sign.split('_')
            videoName = sign[1]
            videoPath = os.path.join(self.VideoStoreDirectory, videoName)
            for file in os.listdir(videoPath):
                absoluteVideoPath = os.path.join(videoPath, file)
                videoList.append(absoluteVideoPath)

        return videoList

    def getVideoName(self,absoluteVideoPath):
        videoName = absoluteVideoPath.split('\\')
        videoName = videoName[5]
        videoName = videoName.split('.')
        videoName = videoName[0]
        return videoName

    def createKeyPointsSubSubDirectory(self, absoluteVideoPath):
        videoName = self.getVideoName(absoluteVideoPath)
        keypointsSubdirectory = videoName.split('_')[0]

        keypointsSubdirectory = os.path.join(self.KeyPointsStoreDirectory, keypointsSubdirectory)
        if not os.path.exists(keypointsSubdirectory):
            os.mkdir(keypointsSubdirectory)

        keypointsSubSubdirectory = os.path.join(keypointsSubdirectory, videoName)
        if not os.path.exists(keypointsSubSubdirectory):
            os.mkdir(keypointsSubSubdirectory)

        return (keypointsSubSubdirectory, videoName)

    def createImageSubSubDirectory(self,absoluteVideopath):
        videoName = self.getVideoName(absoluteVideopath)
        imageSubdirectory = videoName.split('_')[0]

        imageSubdirectory = os.path.join(self.ImageStoreDirectory, imageSubdirectory)
        if not os.path.exists(imageSubdirectory):
            os.mkdir(imageSubdirectory)

        imageSubSubdirectory = os.path.join(imageSubdirectory, videoName)
        if not os.path.exists(imageSubSubdirectory):
            os.mkdir(imageSubSubdirectory)

        return (imageSubSubdirectory, videoName)