import numpy as np
import cv2.aruco as aruco
import cv2
import glob2 as glob
import xlwt
import os
import math
import pandas as pd
def calibration():
   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
   objp = np.zeros((6 * 7, 3), np.float32)
   objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
   objp = objp * 24.2  # 18.1 mm

   objpoints = []  # 3d point in real world space
   imgpoints = []  # 2d points in calib_image plane.

   images = glob.glob('calib_image/*.jpg')
   for fname in images:
      img = cv2.imread(fname)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

      if ret == True:
         objpoints.append(objp)

         corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
         imgpoints.append(corners2)

         img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
         # cv2.imshow('img', img)
         # cv2.waitKey(1000)

   # cv2.destroyAllWindows()

   img = cv2.imread(images[0])
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

   return mtx, dist, rvecs, tvecs


def SquareFinder(img):
    limitCosine = 0.6
    minArea = 100
    maxArea = 1000
    maxError = 0.025



    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    for i in range(len(contours)):
        arclen = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], arclen*maxError, True)
        # print(len(approx))
        if len(approx) == 4 and abs(cv2.contourArea(approx)) > minArea and abs(cv2.contourArea(approx)) < maxArea and cv2.isContourConvex(approx):
            approx = np.squeeze(approx, axis=1)
            maxCos = 0
            for j in range(2, 5):
                cos = abs(angleCornerPointsCos(approx[j%4], approx[j-2], approx[j-1]))
                maxCosine = max(maxCos, cos)

            if maxCosine < limitCosine:
                return approx


def angleCornerPointsCos(b, c, a):
    dx1 = b[0] - a[0]
    dy1 = b[1] - a[1]
    dx2 = c[0] - a[0]
    dy2 = c[1] - a[1]
    l = math.sqrt(dx1*dx1 + dy1*dy1) * math.sqrt(dx2*dx2 + dy2*dy2)
    return (dx1*dx2 + dy1*dy2) / (l + 1e-12)


def deeperAruco():
    cap = cv2.VideoCapture(0)

    while (True):
        ret, frame = cap.read()
        if ret == True:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)

            approx = SquareFinder(img)
            if approx is not None:
                pts = approx.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 255))
                cv2.imshow("img1", img)  # displayd

                cv2.imshow("img2", frame)  # displayd
            else:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
                break

    cap.release()  # When everything done, release the capture
    cv2.destroyAllWindows()

def showResult():
    camera_matrix, dist_coeffs, rvecs, tvecs = calibration()
    cap = cv2.VideoCapture(0)


    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    squareLength = 1.67
    markerLength = 1
    arucoParams = aruco.DetectorParameters_create()


    while(True):
        ret, frame = cap.read()

        if ret == True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=arucoParams)
            #
            # print('c',corners)
            # print('ids', ids)
            if np.any(ids != None):

                # for corner, id in zip(corners, ids):
                image = aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))

                contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # if np.any(ids != None):
            #     diamondCorners, diamondIds = aruco.detectCharucoDiamond(frame_gray, corners, ids,
            #                                                             squareLength / markerLength)
            #     if len(diamondCorners) >= 1:
            #         im_with_diamond = aruco.drawDetectedDiamonds(frame, diamondCorners, diamondIds, (0, 255, 0))
            #         rvec, tvec, _ = aruco.estimatePoseSingleMarkers(diamondCorners, squareLength, camera_matrix,
            #                                                         dist_coeffs)  # posture estimation from a diamond
            #
            #
            #         im_with_diamond = aruco.drawAxis(im_with_diamond, camera_matrix, dist_coeffs, rvec, tvec, 1)
            #
            #
            else:
                image = frame

            cv2.imshow("img", image)  # display

            if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
                break

    cap.release()  # When everything done, release the capture
    cv2.destroyAllWindows()

# showResult()
deeperAruco()