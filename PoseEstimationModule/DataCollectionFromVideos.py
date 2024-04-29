import os
import DirectorySetup
from PoseEstimationClass import PoseEstimation
import cv2
import numpy as np
from Utilities import Utilities

def extractKeyPointsFromVideos():
    utilities = Utilities()
    numberOfFrames = 30
    videoList = utilities.getVideoList()
    poseEstimation = PoseEstimation()

    for video in videoList:
        #Get the video name and keypoints subdirectory
        keypointsSubSubDirectory, videoName = utilities.createKeyPointsSubSubDirectory(video)
        imageSubSubDirectory, videoName = utilities.createImageSubSubDirectory(video)

        capture = cv2.VideoCapture(video)
        count = 0
        while capture.isOpened():
            success, frame = capture.read()
            # Extract KeyPoints
            if not success or count == numberOfFrames:
                break
            else:
                count += 1
                keypoints = poseEstimation.extractKeyPoints(frame)

                keypointsPath = os.path.join(keypointsSubSubDirectory, videoName+'_'+str(count))
                imagePath = os.path.join(imageSubSubDirectory, videoName + '_' + str(count)+'.jpg')

                np.save(keypointsPath, keypoints)

                imagePath = os.path.join(imagePath)
                cv2.imwrite(imagePath, frame)

                cv2.imshow('VideoFeed', frame)

                key = cv2.waitKey(10)
                if key == ord('q'):
                    break

        capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    extractKeyPointsFromVideos()


