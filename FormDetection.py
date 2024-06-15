import cv2
import numpy as np
import time
import PoseModule as pm

def mainFunction():
    cap = cv2.VideoCapture("uploads/formDetection.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

    while True:
        success, img = cap.read()
        if not success:
            break

        # Perform pose detection
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            angle = detector.findAngle(img, 12, 14, 16)
            per = np.interp(angle, (60, 160), (100,0))
            bar = np.interp(angle, (60, 160), (100, 650))

            # Check for the dumbbell curls
            if per == 100:
                color = (0,0,255)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,255,0)
                if dir == 1:
                    count += 0.5
                    dir = 0

            # Draw Bar
            cv2.rectangle(img, (150,150), (0, 0), color, cv2.FILLED)
            cv2.rectangle(img, (int(bar),100), (0, 0), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (20, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0), 4)

            # Draw Curl Count
            cv2.putText(img, f'{count}', (0, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # Write the frame to the output video
        out.write(img)
                
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

mainFunction()
