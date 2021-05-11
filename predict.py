import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, models, transforms
import cv2.aruco as aruco
import cv2
from main import calibration
import numpy as np
from torch.autograd import Variable
from model import CNN_Model1, CNN_Model2


from PIL import Image

def showResult():
    camera_matrix, dist_coeffs, rvecs, tvecs = calibration()
    cap = cv2.VideoCapture(0)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    squareLength = 1.67
    markerLength = 1
    arucoParams = aruco.DetectorParameters_create()

    test_transforms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path_r = 'file/model_r.ckpt'
    model1 = CNN_Model1(model=1).to(device)
    model1.load_state_dict(torch.load(model_path_r))
    model1.eval()

    model_path_t = 'file/model_t.ckpt'
    model2 = CNN_Model2(model=2).to(device)
    model2.load_state_dict(torch.load(model_path_t))
    model2.eval()

    while(True):
        ret, frame = cap.read()

        error = [0, 0]
        if ret == True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # PIL_frame = Image.fromarray(frame_gray)
            PIL_frame = Image.fromarray(frame)
            image_tensor = test_transforms(PIL_frame)
            image_tensor = image_tensor.unsqueeze_(0)
            input = Variable(image_tensor)
            input = input.to(device)
            output1 = model1(input)
            output2 = model2(input)


            corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=arucoParams)

            if np.any(ids != None):
                diamondCorners, diamondIds = aruco.detectCharucoDiamond(frame_gray, corners, ids,
                                                                        squareLength / markerLength)
                if len(diamondCorners) >= 1:
                    im_with_diamond = aruco.drawDetectedDiamonds(frame, diamondCorners, diamondIds, (0, 255, 0))
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(diamondCorners, squareLength, camera_matrix,
                                                                    dist_coeffs)  # posture estimation from a diamond
                    pr_array = output1.cpu().data.numpy()
                    p_rvec = pr_array[0]
                    p_rvec = p_rvec.reshape((1, 1, -1))

                    pt_array = output2.cpu().data.numpy()
                    p_tvec = pt_array[0]
                    p_tvec = p_tvec.reshape((1, 1, -1))

                    error[0] = ((p_rvec[0][0] - rvec[0][0]) ** 2).mean()
                    error[1] = ((p_tvec[0][0] - tvec[0][0]) ** 2).mean()


                    im_with_diamond = aruco.drawAxis(im_with_diamond, camera_matrix, dist_coeffs, rvec, tvec, 1)
                    im_with_diamond = aruco.drawAxis(im_with_diamond, camera_matrix, dist_coeffs, p_rvec, p_tvec, 1)
                    cv2.putText(im_with_diamond, str(error), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

            else:
                im_with_diamond = frame

            cv2.imshow("diamondLeft", im_with_diamond)  # display

            if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
                break

    cap.release()  # When everything done, release the capture
    cv2.destroyAllWindows()

if __name__ == '__main__':
    showResult()