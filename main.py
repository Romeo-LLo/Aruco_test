import numpy as np
import cv2.aruco as aruco
import cv2
import glob2 as glob
import xlwt
import os
import pandas as pd

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

def detection():
   camera_matrix, dist_coeffs, rvecs, tvecs = calibration()
   cap = cv2.VideoCapture(0)

   aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
   squareLength = 1.67
   markerLength = 1
   arucoParams = aruco.DetectorParameters_create()

   workbook = xlwt.Workbook(encoding='ascii')
   tr_worksheet = workbook.add_sheet('Train Worksheet')
   tt_worksheet = workbook.add_sheet('Test Worksheet')

   style = xlwt.XFStyle()  # 初始化樣式
   font = xlwt.Font()  # 為樣式建立字型
   tr_path = './train_image/'
   tt_path = './test_image/'

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

               if (count % 2) == 0:
                  if count == 0:
                     id = 0
                  else:
                     id = int(count/2)
                  print("Train Recorded no.", id)
                  tr_worksheet.write(id, 0, id)
                  tr_worksheet.write(id, 1, rvec[0][0][0])
                  tr_worksheet.write(id, 2, rvec[0][0][1])
                  tr_worksheet.write(id, 3, rvec[0][0][2])
                  tr_worksheet.write(id, 4, tvec[0][0][0])
                  tr_worksheet.write(id, 5, tvec[0][0][1])
                  tr_worksheet.write(id, 6, tvec[0][0][2])
                  # cv2.imwrite(os.path.join(tr_path, '{}.jpg'.format(id)), frame_gray)
                  cv2.imwrite(os.path.join(tr_path, '{}.jpg'.format(id)), frame)


               else:
                  if count == 1:
                     id = 0
                  else:
                     id = int(count / 2)
                  print("Test Recorded no.", id)
                  tt_worksheet.write(id, 0, id)
                  tt_worksheet.write(id, 1, rvec[0][0][0])
                  tt_worksheet.write(id, 2, rvec[0][0][1])
                  tt_worksheet.write(id, 3, rvec[0][0][2])
                  tt_worksheet.write(id, 4, tvec[0][0][0])
                  tt_worksheet.write(id, 5, tvec[0][0][1])
                  tt_worksheet.write(id, 6, tvec[0][0][2])
                  cv2.imwrite(os.path.join(tt_path, '{}.jpg'.format(id)), frame)
               count += 1
               im_with_diamond = aruco.drawAxis(im_with_diamond, camera_matrix, dist_coeffs, rvec, tvec, 1)
         else:
            im_with_diamond = frame

         cv2.imshow("diamondLeft", im_with_diamond)  # display

         if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
            # workbook.save('train_data.xls')
            workbook.save('data.xls')

            break
      else:
         workbook.save('data.xls')
         break
   cap.release()  # When everything done, release the capture
   cv2.destroyAllWindows()

def toCSV():
   # read_file = pd.read_excel(r'train_data.xls')
   # read_file.to_csv(r'train_data.csv', index=None, header=True)
   # read_file = pd.read_excel(r'test_data.xls')
   # read_file.to_csv(r'test_data.csv', index=None, header=True)
   train_file = pd.read_excel(r'data.xls', sheet_name='Train Worksheet')
   test_file = pd.read_excel(r'data.xls', sheet_name='Test Worksheet')

   train_file.to_csv(r'train_data.csv', index=None, header=True)
   test_file.to_csv(r'test_data.csv', index=None, header=True)


if __name__ == '__main__':
   # Aruco_creation()
   # static_detect()
   # calibration()
   # detection()
   # toCSV()
   pass
