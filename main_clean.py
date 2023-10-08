# Import required libraries
from utils_clean import *
import cv2
import numpy as np

# Initialize variables
w, h = 360, 240  # Image width and height
center = w // 2  # Center of the image
pid_yaw = [0.7, 0.5, 0]  # PID gains for yaw
pid_fb = [0.7, 0.5, 0]  # PID gains for forward/backward
pid_ud = [0.7, 0.5, 0]  # PID gains for up/down
pError_yaw, pError_fb, pError_ud = 0, 0, 0  # Initialize previous errors
facearea = 0  # Current face area
desiredfacearea = 5000  # Target face area
startCounter = 0  # Takeoff flag

# Initialize drone
myDrone = initializeTello()

# Main loop
while True:

    # Drone takeoff if it hasn't already
    if startCounter == 0:
        myDrone.takeoff()
        startCounter = 1

    # Get video frame and face information
    img = telloGetFrame(myDrone, w, h)
    img, info = findFace(img)
    facearea = findArea(img)

    # Handle keyboard inputs for modifying desired face area and center
    if cv2.waitKey(10) & 0xFF == ord('w'):
        pError_fb = 0
        desiredfacearea += 100
        print('desiredfacearea +10')
    if cv2.waitKey(10) & 0xFF == ord('s'):
        pError_fb = 0
        desiredfacearea -= 100
        print('desiredfacearea -10')
    if cv2.waitKey(10) & 0xFF == ord('a'):
        pError_yaw = 0
        center += 10
        print('center +10')
    if cv2.waitKey(10) & 0xFF == ord('d'):
        pError_yaw = 0
        center -= 10
        print('center -10')

    # Constrain desired face area and center within limits
    desiredfacearea = int(np.clip(desiredfacearea, 3000, 9000))
    center = int(np.clip(center, 0, 360))

    # Update drone's position to track face
    pError_yaw, pError_fb, pError_ud = trackFace(
        myDrone, info, w, pid_yaw, pid_fb, pError_yaw, pError_fb, facearea, h, 
        pError_ud, pid_ud, center, desiredfacearea
    )

    # Display video feed
    cv2.imshow('Image', img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        myDrone.land()
        break
