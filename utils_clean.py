from djitellopy import Tello  # Import Tello drone library
import cv2  # Import OpenCV for image processing
import numpy as np  # Import numpy for numerical operations

def initializeTello():
    """
    Initialize and return a Tello drone object with default settings.

    Returns:
        myDrone (Tello object): Initialized Tello drone object
    """
    myDrone = Tello()  # Create a new Tello drone object
    
    myDrone.connect()  # Connect to the drone
    
    # Initialize drone velocities and speed to zero
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0
    
    # Display drone battery level
    print(f"Drone Battery: {myDrone.get_battery()}%")
    
    # Turn off and on the video stream for initial setup
    myDrone.streamoff()
    myDrone.streamon()
    
    return myDrone  # Return the initialized drone object


def telloGetFrame(myDrone, w=360, h=240):
    """
    Get and return a video frame from the Tello drone.
    The frame is resized to the dimensions specified by parameters w and h.
    
    Parameters:
        myDrone (Tello object): The drone object from which to capture the frame.
        w (int, optional): The desired width of the output frame. Default is 360.
        h (int, optional): The desired height of the output frame. Default is 240.
    
    Returns:
        img (numpy.ndarray): The captured and resized video frame.
    """
    myFrame = myDrone.get_frame_read()  # Read the current frame from the drone's camera
    myFrame = myFrame.frame  # Extract the frame from the frame reader
    img = cv2.resize(myFrame, (w, h))  # Resize the frame to the specified dimensions
    return img  # Return the resized frame

def findFace(img):
    """
    Detects the largest face in the given image and returns the image with a rectangle drawn around the face.
    Also returns the center coordinates and area of the detected face.
    
    Parameters:
        img (numpy.ndarray): The input image in which to detect faces.
    
    Returns:
        img (numpy.ndarray): The input image with the largest face highlighted using a rectangle.
        list: A list containing the coordinates of the center of the largest face and its area. 
              Returns [[0, 0], 0] if no faces are detected.
    """
    # Initialize OpenCV's face detector
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 6)

    # Initialize lists to store the center coordinates and areas of detected faces
    myFaceListC = []
    myFaceListArea = []

    # Loop through the list of detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around each face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Calculate the center coordinates of the face
        cx = x + w // 2
        cy = y + h // 2

        # Calculate the area of the face
        area = w * h

        # Add the center coordinates and area to their respective lists
        myFaceListArea.append(area)
        myFaceListC.append([cx, cy])

    # If at least one face is detected, find the largest one
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def findArea(img):
    """
    Detects the largest face in the given image and returns its area.
    
    Parameters:
        img (numpy.ndarray): The input image in which to detect faces.
    
    Returns:
        int: The area of the largest detected face. Returns 0 if no faces are detected.
    """
    # Initialize OpenCV's face detector
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 6)

    # Initialize lists to store the center coordinates and areas of detected faces
    myFaceListC = []
    myFaceListArea = []

    # Loop through the list of detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around each face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Calculate the center coordinates of the face
        cx = x + w // 2
        cy = y + h // 2

        # Calculate the area of the face
        area = w * h

        # Add the center coordinates and area to their respective lists
        myFaceListArea.append(area)
        myFaceListC.append([cx, cy])

    # If at least one face is detected, find the largest one
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return myFaceListArea[i]
    else:
        return 0


def trackFace(myDrone, info, w, pid_yaw, pid_fb,  pError_yaw, pError_fb, facearea, h, pError_ud, pid_ud, center, desiredfacearea):
    """
    Tracks a face detected in the frame and adjusts the drone's position accordingly.
    
    Parameters:
        myDrone (Tello): Drone object for controlling the Tello drone.
        info (list): Information about the face [center coordinates, area].
        w (int): Width of the frame.
        pid_yaw, pid_fb, pid_ud (list): PID controller gains for yaw, forward-backward, and up-down.
        pError_yaw, pError_fb, pError_ud (float): Previous errors for yaw, forward-backward, and up-down.
        facearea (int): Area of the detected face.
        h (int): Height of the frame.
        center (int): Horizontal center of the frame.
        desiredfacearea (int): The desired area of the face in the frame for tracking.

    Returns:
        tuple: New error values for yaw, forward-backward, and up-down movements.
    """
    
    # If info list does not contain enough data, return previous errors
    if len(info) < 2:
        return pError_yaw, pError_fb, pError_ud
    
    # Compute the yaw error and speed
    error_yaw = info[0][0] - center
    speed_yaw = pid_yaw[0] * error_yaw + pid_yaw[1] * (error_yaw - pError_yaw)
    speed_yaw = int(np.clip(speed_yaw, -60, 60))
    
    # Compute the forward-backward error and speed
    error_fb = facearea - desiredfacearea
    speed_fb = pid_fb[0] * error_fb + pid_fb[1] * (error_fb - pError_fb)
    speed_fb = int(np.clip(speed_fb, -20, 20))

    # Compute the up-down error and speed
    error_ud = info[0][1] - h // 2
    speed_ud = pid_ud[0] * error_ud + pid_ud[1] * (error_ud - pError_ud) 
    speed_ud = int(np.clip(speed_ud, -25, 25)) #
    
    # Log the speeds and settings
    print(speed_yaw, speed_fb, speed_ud)
    print(center, desiredfacearea)
    
    # Update the drone's velocities based on the calculated speeds
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
        # If no face detected, stop the drone
        myDrone.for_back_velocity = 0
        myDrone.left_right_velocity = 0
        myDrone.up_down_velocity = 0
        myDrone.yaw_velocity = 0

    # Send the updated controls to the drone
    if myDrone.send_rc_control:
        myDrone.send_rc_control(myDrone.left_right_velocity,
                                myDrone.for_back_velocity,
                                myDrone.up_down_velocity,
                                myDrone.yaw_velocity)
    
    return error_yaw, error_fb, error_ud # Return the new error values