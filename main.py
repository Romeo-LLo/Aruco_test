import numpy as np
import cv2.aruco as aruco
import cv2
import glob2 as glob

def Aruco_creation():
   dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
   markerImage = np.zeros((200, 200), dtype=np.uint8)
   markerImage = cv2.aruco.drawMarker(dictionary, 22, 200, markerImage, 1)
   cv2.imwrite("marker22.png", markerImage)

def static_detect(image):
   frame = cv2.imread(image)
   frame = cv2.resize(frame, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
   parameters = aruco.DetectorParameters_create()
   corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
   aruco.drawDetectedMarkers(frame, corners, ids)

   cv2.imshow("frame", frame)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

def calibration():
   # termination criteria
   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

   # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
   objp = np.zeros((6 * 7, 3), np.float32)
   objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

   # Arrays to store object points and image points from all the images.
   objpoints = []  # 3d point in real world space
   imgpoints = []  # 2d points in image plane.

   images = glob.glob(r'/home/user/PycharmProjects/Aruco_test/image/*.jpg')

   for i in images:
     pass
   #
   # for fname in images:
   #    img = cv2.imread(fname)
   #    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   #
   #    # Find the chess board corners
   #    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
   #
   #    # If found, add object points, image points (after refining them)
   #    if ret == True:
   #       objpoints.append(objp)
   #
   #       corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
   #       imgpoints.append(corners2)
   #
   #       # Draw and display the corners
   #       img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
   #       cv2.imshow('img', img)
   #       cv2.waitKey(500)
   #
   # cv2.destroyAllWindows()


if __name__ == '__main__':
   # Aruco_creation()
   # static_detect()
   calibration()