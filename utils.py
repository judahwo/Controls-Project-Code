from djitellopy import Tello
import cv2
import numpy as np


def initializeTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0
    print(myDrone.get_battery())
    myDrone.streamoff()
    myDrone.streamon()
    return myDrone

def telloGetFrame(myDrone, w=360, h=240):
    myFrame = myDrone.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    return img

def findFace(img):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 6)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        myFaceListArea.append(area)
        myFaceListC.append([cx, cy])

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

def findArea(img):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 6)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        myFaceListArea.append(area)
        myFaceListC.append([cx, cy])

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return myFaceListArea[i]
    else:
        return 0


def trackFace(myDrone, info, w, pid_yaw, pid_fb,  pError_yaw, pError_fb, facearea, h, pError_ud, pid_ud, center, desiredfacearea):
    if len(info) < 2:
        return pError_yaw, pError_fb

    error_yaw = info[0][0] - center
    speed_yaw = pid_yaw[0] * error_yaw + pid_yaw[1] * (error_yaw - pError_yaw)
    speed_yaw = int(np.clip(speed_yaw, -60, 60))

    error_fb = facearea - desiredfacearea
    speed_fb = pid_fb[0] * error_fb + pid_fb[1] * (error_fb - pError_fb)
    speed_fb = int(np.clip(speed_fb, -20, 20))

    error_ud = info[0][1] - h // 2
    speed_ud = pid_ud[0] * error_ud + pid_ud[1] * (error_ud - pError_ud)
    speed_ud = int(np.clip(speed_ud, -25, 25))

    print(speed_yaw, speed_fb, speed_ud)
    print(center, desiredfacearea)

    if info[0][0] != 0:
        if abs(speed_yaw) < 20:
            myDrone.left_right_velocity = speed_yaw
            myDrone.yaw_velocity = 0
        else:
            myDrone.yaw_velocity = speed_yaw
            myDrone.left_right_velocity = 0
        myDrone.for_back_velocity = -speed_fb
        myDrone.up_down_velocity = -speed_ud
    else:
        myDrone.for_back_velocity = 0
        myDrone.left_right_velocity = 0
        myDrone.up_down_velocity = 0
        myDrone.yaw_velocity = 0
        error_yaw = 0
        error_fb = 0
        error_ud = 0
    if myDrone.send_rc_control:
        myDrone.send_rc_control(myDrone.left_right_velocity,
                                myDrone.for_back_velocity,
                                myDrone.up_down_velocity,
                                myDrone.yaw_velocity)
    return error_yaw, error_fb, error_ud