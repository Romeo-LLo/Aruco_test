import torch
from torchvision import datasets, models, transforms
import cv2.aruco as aruco
import cv2
from main import calibration
import numpy as np
from torch.autograd import Variable
from main import calibration
import os
from PIL import Image


def video_label():
   camera_matrix, dist_coeffs, rvecs, tvecs = calibration()
   cap = cv2.VideoCapture(0)

   aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
   squareLength = 1.67
   markerLength = 1
   arucoParams = aruco.DetectorParameters_create()

   count = 0
   while (True):
      ret, frame = cap.read()
      if ret == True:
         # frame_remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)  # for fisheye remapping
         # frame_remapped_gray = cv2.cvtColor(frame_remapped, cv2.COLOR_BGR2GRAY)
         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

         corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=arucoParams)

         if np.any(ids != None):
            diamondCorners, diamondIds = aruco.detectCharucoDiamond(frame_gray, corners, ids, squareLength / markerLength)
            if len(diamondCorners) >= 1:
               im_with_diamond = aruco.drawDetectedDiamonds(frame, diamondCorners, diamondIds, (0, 255, 0))
               rvec, tvec, _ = aruco.estimatePoseSingleMarkers(diamondCorners, squareLength, camera_matrix, dist_coeffs)  # posture estimation from a diamond
         else:
            im_with_diamond = frame

         cv2.imshow("diamondLeft", im_with_diamond)  # display

         if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
            break
      else: break
   cap.release()  # When everything done, release the capture
   cv2.destroyAllWindows()

def image_label():
    camera_matrix, dist_coeffs, rvecs, tvecs = calibration()

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    squareLength = 1.67
    markerLength = 1
    arucoParams = aruco.DetectorParameters_create()

    for frame in os.listdir()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=arucoParams)

    if np.any(ids != None):
       diamondCorners, diamondIds = aruco.detectCharucoDiamond(frame_gray, corners, ids, squareLength / markerLength)
       if len(diamondCorners) >= 1:
           im_with_diamond = aruco.drawDetectedDiamonds(frame, diamondCorners, diamondIds, (0, 255, 0))
           rvec, tvec, _ = aruco.estimatePoseSingleMarkers(diamondCorners, squareLength, camera_matrix,
                                                           dist_coeffs)  # posture estimation from a diamond
    else:
       im_with_diamond = frame

    cv2.imshow("diamondLeft", im_with_diamond)  # display

