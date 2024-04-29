import cv2
from PoseEstimationClass import PoseEstimation


def main():
    poseEstimation = PoseEstimation()
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        success, frame = capture.read()
        # frame = cv2.cvt
        results = poseEstimation.extractKeyPoints(frame)
        # print(results)
        cv2.imshow('Video Feed', frame)
        key = cv2.waitKey(100)
        if key == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

    cv2.imshow('Last Image', frame)
    cv2.waitKey(0)
    print(results.shape)

if __name__ == '__main__':
    main()