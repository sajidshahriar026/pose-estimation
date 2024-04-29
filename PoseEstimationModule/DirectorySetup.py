import os

baseDirectory = 'E:\Thesis\pythonProject'
keyPointsStoreDirectory = 'Keypoints'
imageStoreDirectory = 'Images'
videoStoreDirectory = 'Videos'
signListDirectory = 'SignList'
logDirectory = 'Log Directory'

KeyPointsStoreDirectory = os.path.join(baseDirectory, keyPointsStoreDirectory)
ImageStoreDirectory = os.path.join(baseDirectory, imageStoreDirectory)
VideoStoreDirectory = os.path.join(baseDirectory, videoStoreDirectory)
SignListDirectory = os.path.join(baseDirectory, signListDirectory)
LogDirectory = os.path.join(baseDirectory, logDirectory)

DataDirectories = [baseDirectory, ImageStoreDirectory, VideoStoreDirectory, SignListDirectory,LogDirectory]


if __name__ == '__main__':
    for directory in DataDirectories:
        if not os.path.exists(directory):
            os.mkdir(directory)


