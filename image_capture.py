import cv2

camera = cv2.VideoCapture(0)
i = 0
while True:
    (grabbed, img) = camera.read()
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('j'):
        i += 1
        u = str(i)
        filename=str('/home/user/PycharmProjects/Aruco_test/image/image/img' + u + '.jpg')
        cv2.imwrite(filename, img)
        print( '寫入：', filename)
        cv2.imshow('img_out', filename)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
