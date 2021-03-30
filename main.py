import numpy as np
import cv2.aruco as aruco
import cv2
import glob2 as glob
import xlwt
import os

def Aruco_creation():
   dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

   markerImage = np.zeros((200, 200), dtype=np.uint8)
   markerImage = cv2.aruco.drawMarker(dictionary, 22, 200, markerImage, 1)
   cv2.imwrite("marker22.png", markerImage)

def Diamond_Aruco_creation():
   pass

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
   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
   objp = np.zeros((6 * 7, 3), np.float32)
   objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
   objp = objp * 24.2  # 18.1 mm

   objpoints = []  # 3d point in real world space
   imgpoints = []  # 2d points in image plane.

   images = glob.glob('image/*.jpg')
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

def detection():
   camera_matrix, dist_coeffs, rvecs, tvecs = calibration()
   cap = cv2.VideoCapture(0)

   aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
   squareLength = 1.67
   markerLength = 1
   arucoParams = aruco.DetectorParameters_create()

   workbook = xlwt.Workbook(encoding='ascii')
   worksheet = workbook.add_sheet('My Worksheet')
   style = xlwt.XFStyle()  # 初始化樣式
   font = xlwt.Font()  # 為樣式建立字型
   path = './train_image/'
   count = 1
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


               if ((count % 3) == 0):
                  id = int(count/3)
                  print("Recorded no.", id)
                  worksheet.write(id, 0, id+1)
                  worksheet.write(id, 1, rvec[0][0][0])
                  worksheet.write(id, 2, rvec[0][0][1])
                  worksheet.write(id, 3, rvec[0][0][2])
                  worksheet.write(id, 4, tvec[0][0][0])
                  worksheet.write(id, 5, tvec[0][0][1])
                  worksheet.write(id, 6, tvec[0][0][2])
                  cv2.imwrite(os.path.join(path, '{}.jpg'.format(id)), frame_gray)
               count += 1
               im_with_diamond = aruco.drawAxis(im_with_diamond, camera_matrix, dist_coeffs, rvec, tvec, 1)  # axis length 100 can be changed according to your requirement
         else:
            im_with_diamond = frame

         cv2.imshow("diamondLeft", im_with_diamond)  # display

         if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
            workbook.save('train_data.xls')
            break
      else:
         workbook.save('train_data.xls')
         break
   cap.release()  # When everything done, release the capture
   cv2.destroyAllWindows()

if __name__ == '__main__':
   # Aruco_creation()
   # static_detect()
   # calibration()
   detection()