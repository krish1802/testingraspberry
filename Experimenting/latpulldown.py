import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("latpull.mp4")

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    # img = cv2.imread("test.jpg")
    # img.resize((1280, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    if len(lmList) != 0:
        # #Right Arm
        angle = detector.findAngle(img, 12, 14, 16)
        #Left Arm
        right_angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (60, 160), (100,0))
        bar = np.interp(angle, (60, 160), (100, 650))
        #print(per)

        # Check for the dumbbell curls
        if per == 100:
            color = (0,0,255)
            if dir ==0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0,255,0)
            if dir == 1:
                count += 0.5
                dir = 0

        # Draw Bar
        # cv2.rectangle(img, (150,150), (0, 0), color, cv2.FILLED)
        # cv2.rectangle(img, (int(bar),100), (0, 0), color, cv2.FILLED)
        # cv2.putText(img, f'{int(per)} %', (20, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0), 4)

        # Draw Curl Count
        #cv2.rectangle(img, (0,100), (120, 50), (0, 255, 0), cv2.FILLED)
        #cv2.putText(img, f'{count}', (0, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

            
    cTime = time.time()
    fps = 1/(cTime -pTime)
    pTime = cTime
    #cv2.putText(img, f'{fps}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)