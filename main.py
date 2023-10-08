from utils import *
import cv2

w, h = 360, 240
center = w // 2

pid_yaw = [0.7, 0.7, 0]
pid_fb = [0.1, 0.1, 0]
pid_ud = [0.556, 0, 0]
pError_yaw = 0
pError_fb = 0
pError_ud = 0
facearea=0
desiredfacearea=5000
startCounter = 0

myDrone = initializeTello()

while True:

    if startCounter == 0:
        myDrone.takeoff()
        startCounter = 1

    img = telloGetFrame(myDrone, w, h)
    img, info = findFace(img)
    facearea=findArea(img)

    if cv2.waitKey(10) & 0xFF == ord('w'):
        pError_fb = 0
        desiredfacearea = desiredfacearea + 100
        print('desiredfacearea +10')
    if cv2.waitKey(10) & 0xFF == ord('s'):
        pError_fb = 0
        desiredfacearea = desiredfacearea - 100
        print('desiredfacearea -10')
    if cv2.waitKey(10) & 0xFF == ord('a'):
        pError_yaw = 0
        center = center + 10
        print('center +10')
    if cv2.waitKey(10) & 0xFF == ord('d'):
        pError_yaw = 0
        center = center - 10
        print('center -10')

    desiredfacearea = int(np.clip(desiredfacearea, 3000, 9000))
    center = int(np.clip(center, 0, 360))

    pError_yaw, pError_fb, pError_ud = trackFace(myDrone, info, w, pid_yaw, pid_fb,  pError_yaw, pError_fb, facearea, h, pError_ud, pid_ud, center, desiredfacearea)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        myDrone.land()
        break