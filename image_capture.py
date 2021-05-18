import cv2
import os

camera = cv2.VideoCapture(0)
i = 0
while True:
    (grabbed, img) = camera.read()
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('z'):
        i += 1
        u = str(i)
        path = 'home/user/PycharmProjects/Aruco_test/calib_image/'
        image_name = 'img{}.jpg'.format(u)
        # cv2.imwrite(os.path.join(path + image_name), img)
        cv2.imwrite(image_name, img)
        print('寫入：', image_name)
        # if not cv2.imwrite(image_name, img):
        #     raise Exception("Could not write calib_image")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
