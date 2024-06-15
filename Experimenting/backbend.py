import cv2
import numpy as np
import time
import PoseModule as pm
import math

def calculate_backbend_angle(lmList):
    # Function to calculate the backbend angle
    # This function calculates the angle between the vectors representing the upper body and lower body
    
    # Indexes of relevant keypoints
    right_shoulder = lmList[14]  # Right shoulder
    right_hip = lmList[24]       # Right hip  
    spine = lmList[12]           # Middle of the spine

    # Calculate the vectors representing the upper body (shoulder to spine) and the lower body (hip to spine)
    upper_body_vector = [right_shoulder[1] - spine[1], right_shoulder[2] - spine[2]]  # [y_diff, x_diff]
    lower_body_vector = [right_hip[1] - spine[1], right_hip[2] - spine[2]]                # [y_diff, x_diff]

    # Calculate the dot product of the two vectors
    dot_product = upper_body_vector[0] * lower_body_vector[0] + upper_body_vector[1] * lower_body_vector[1]

    # Calculate the magnitudes of the vectors
    upper_body_magnitude = math.sqrt(upper_body_vector[0] ** 2 + upper_body_vector[1] ** 2)
    lower_body_magnitude = math.sqrt(lower_body_vector[0] ** 2 + lower_body_vector[1] ** 2)

    # Calculate the cosine of the angle between the vectors using the dot product formula
    cos_angle = dot_product / (upper_body_magnitude * lower_body_magnitude)

    # Calculate the angle in radians using the arccosine function
    angle_rad = math.acos(cos_angle)

    # Convert the angle from radians to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def deadlift():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"backbendWrong.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0

    while True:
        success, img = cap.read()
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) != 0:
            # Calculate backbend angle
            backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if backbend_angle > 55:
                # Show "backbend" prompt
                cv2.putText(img, 'Backbend!', (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                
                # You can add further actions here when backbend is detected
            
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
deadlift()