# This file contains all the functions used in the main file 

# Import Libraries
from djitellopy import Tello  
import cv2  
import numpy as np


def initializeTello():
    """Initialize Tello drone and return the object.

    param: None

    return: Tello object
    """
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = myDrone.left_right_velocity = myDrone.up_down_velocity = myDrone.yaw_velocity = myDrone.speed = 0
    print(f"Drone Battery: {myDrone.get_battery()}%")
    myDrone.streamoff()
    myDrone.streamon()
    return myDrone

def telloGetFrame(myDrone, w=360, h=240):
    """Capture and return a resized video frame from Tello drone.

    param: myDrone: Tello object
    param: w: frame width
    param: h: frame height

    return: resized video frame
    """

    frame = myDrone.get_frame_read().frame
    img = cv2.resize(frame, (w, h))
    return img

def findFace(img):
    """Detect and return largest face coordinates and area.

    param: img: video frame

    return: img: frame with rectangle around largest face
    return: [cx, cy]: center of largest face
    return: area: area of largest face
    """

    # Detect face
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 6)
    
    # Get the largest face
    myFaceListC, myFaceListArea = [], []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cx, cy, area = x + w // 2, y + h // 2, w * h
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    
    # Return center and area of the largest face
    if myFaceListArea:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    return img, [[0, 0], 0]

def findArea(img):
    """ Detect and return area of the largest face.

    param: img: video frame
    
    return: area: area of the largest face
    """
    
    # Reusing findFace for getting the area
    info = findFace(img)
    return info[1] if info else 0

def trackFace(myDrone, info, w, pid_yaw, pid_fb, pError_yaw, pError_fb, facearea, h, pError_ud, pid_ud, center, desiredfacearea):
    """Track face and adjust drone position.

    return: pError_yaw: previous error for yaw
    return: pError_fb: previous error for forward-backward
    return: pError_ud: previous error for up-down
    """

    # if no face detected, return previous errors
    if len(info) < 2:
        return pError_yaw, pError_fb, pError_ud
    
    # PID values for yaw
    error_yaw = info[0][0] - center                                                                                     # error for yaw
    speed_yaw = pid_yaw[0] * error_yaw + pid_yaw[1] * (error_yaw - pError_yaw) + pid_yaw[2] * (error_yaw + pError_yaw)  # PID values for yaw
    speed_yaw = int(np.clip(speed_yaw, -60, 60))                                                                        # constrain yaw speed
    
    # PID values for forward-backward
    error_fb = facearea - desiredfacearea                                                                               # error for forward-backward
    speed_fb = pid_fb[0] * error_fb + pid_fb[1] * (error_fb - pError_fb) + pid_fb[2] * (error_fb + pError_fb)           # PD values for forward-backward
    speed_fb = int(np.clip(speed_fb, -20, 20))                                                                          # constrain forward-backward speed

    # PID values for up-down
    error_ud = info[0][1] - h // 2                                                                                      # error for up-down
    speed_ud = pid_ud[0] * error_ud + pid_ud[1] * (error_ud - pError_ud)  + pid_ud[2] * (error_ud + pError_ud)          # PD values for up-down
    speed_ud = int(np.clip(speed_ud, -25, 25))                                                                          # constrain up-down speed
    
    # Update previous errors
    print(speed_yaw, speed_fb, speed_ud)
    print(center, desiredfacearea)
    
    # Update previous errors
    if info[0][0]:
        myDrone.left_right_velocity = speed_yaw if abs(speed_yaw) < 20 else 0
        myDrone.yaw_velocity = speed_yaw if abs(speed_yaw) >= 20 else 0
        myDrone.for_back_velocity = -speed_fb
        myDrone.up_down_velocity = -speed_ud
    else:
        myDrone.for_back_velocity = myDrone.left_right_velocity = myDrone.up_down_velocity = myDrone.yaw_velocity = 0

    # Update previous errors
    if myDrone.send_rc_control:
        myDrone.send_rc_control(myDrone.left_right_velocity, myDrone.for_back_velocity, myDrone.up_down_velocity, myDrone.yaw_velocity)
    
    # Return previous errors
    return error_yaw, error_fb, error_ud
