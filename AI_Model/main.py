import cv2
import numpy as np
import time
import PoseModule as pm
import math
import os

def ensure_unique_file(filename):
    if os.path.exists(filename):
        base, extension = os.path.splitext(filename)
        counter = 1
        new_filename = f"{base}{counter}{extension}"
        while os.path.exists(new_filename):
            counter += 1
            new_filename = f"{base}{counter}{extension}"
        filename = new_filename
    return filename

def hip_instability(timestamps, img):
    filename = ensure_unique_file("hip_angle_output.txt")

    with open(filename, 'w') as file:
        file.write("Your hips were unstable at:\n")
        i = 0
        while i < len(timestamps):
            file.write(f"Time at which your back was unstable: {timestamps[i]} Seconds\n")
            i += 1

def hand_instability(timestamps, img):
    filename = ensure_unique_file("hand_angle_output.txt")

    with open(filename, 'w') as file:
        file.write("Your hands were straight at:\n")
        i = 0
        while i < len(timestamps):
            file.write(f"Time at which your back was unstable: {timestamps[i]} Seconds\n")
            i += 1

def legStraight(timestamps, img):
    filename = ensure_unique_file("knee_straight_output.txt")

    if timestamps[0] == None:
        with open(filename, 'w') as file:
            file.write("Your legs were Straight \n")
    else:
        with open(filename, 'w') as file:
            file.write("Your legs were not straight at:\n")
            i = 0
            while i < len(timestamps):
                file.write(f"Time at which your legs weren't straight: {timestamps[i]} Seconds\n")
                i += 1

def back_bend(timestamps, img):
    filename = ensure_unique_file("back_bend_output.txt")

    with open(filename, 'w') as file:
        file.write("Your back was bend at:\n")
        for i, timestamp in enumerate(timestamps):
            message = f'{time.strftime("%H:%M:%S", time.localtime(timestamp))}\n'
            cv2.putText(img, message, (50, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            file.write(message)

def resize_image(image, width=None, height=None):
    # Resize the image while maintaining aspect ratio
    if width is None and height is None:
        return image
    elif width is not None and height is not None:
        raise ValueError("Only one of 'width' or 'height' should be provided.")
    
    if width is not None:
        aspect_ratio = image.shape[1] / image.shape[0]
        new_height = int(width / aspect_ratio)
        return cv2.resize(image, (width, new_height))
    else:
        aspect_ratio = image.shape[0] / image.shape[1]
        new_width = int(height / aspect_ratio)
        return cv2.resize(image, (new_width, height))

def squat_backbend(lmList):
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

def put_text_in_frame(image, text, position, font, scale, color, thickness):
    """
    Helper function to ensure the text is within the frame.
    """
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    # Ensure the text is within the frame
    x = max(position[0], 0)
    y = max(position[1], text_size[1])
    if x + text_size[0] > image.shape[1]:
        x = image.shape[1] - text_size[0]
    if y + text_size[1] > image.shape[0]:
        y = image.shape[0] - text_size[1]
    return (x, y)

def resize_overlay_image(overlay_img, frame_width, padding=50):
    # Calculate the new width and height while maintaining the aspect ratio
    new_width = frame_width - 2 * padding
    aspect_ratio = overlay_img.shape[0] / overlay_img.shape[1]
    new_height = int(new_width * aspect_ratio)
    
    # Resize the overlay image
    resized_overlay_img = cv2.resize(overlay_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_overlay_img

import psutil


# Arms workout
def bicepCurls(): # THIS ONE IS FROM RHS
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(0)
    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    timestamps = []
    backBend = False
    start_time = time.time()
    counted = False
    timestamp = 0
    ui_width = 170
    lineColor = (0,255,0)
    per = 0
    color = 0
    bar = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        frame_height, frame_width = img.shape[:2]

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 12, 14, 16,lineColor=lineColor)
            hipangle = detector.findAngle(img, 12,24,26, lineColor=lineColor)
            shoudler_angle = detector.findAngle(img, 24, 12, 14, lineColor=lineColor)

            if hipangle > 190:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                text = "KEEP YOUR BACK STRAIGHT"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            elif shoudler_angle < 3:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                text = "KEEP YOUR HANDS STABLE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            timestamps.append(int(timestamp))
            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (80, 150), (100,0))
            bar = np.interp(angle, (80, 150), (100, 700))

            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green

            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 0, 255)
                if dir == 1:
                    count += 0.5
                    dir = 0
        timestamps = list(set(timestamps))
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime


        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        canvas_resized = cv2.resize(canvas, (1920, 1080))
        # if cv2.getWindowProperty("Image with UI", cv2.WND_PROP_VISIBLE) < 1:
        #     break
        cv2.imshow("Image with UI", canvas_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # hip_instability(timestamps, img)

bicepCurls()

def preacherCurls():
    # r is used because windows automatically puts backslashes when copying path, but we need forward slashes in python
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\preachercurls.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    camera_warning_shown = False
    timestamps = []
    handBend = False
    start_time = time.time()
    counted = False
    timestamp = 0
    ui_width = 170
    lineColor = (0,255,0)
    frame_height = 0
    frame_width = 0

    while True:
        success, img = cap.read()
        if not success:
            break
       # img = cv2.imread("test.jpg")
       # img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        frame_height, frame_width = img.shape[:2]

        if len(lmList) != 0:
            #Right Arm 74 Being minimum and 159 being maximum
            angle = detector.findAngle(img, 11, 13, 19)
            #Left Arm
            #angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (220, 300), (0,100))
            bar = np.interp(angle, (220, 300), (100, 650))
            #print(per)

            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green

            # Check for the dumbbell curls
            color = (0,255,0)
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
            
            if angle < 220:
                frame_height, frame_width = img.shape[:2]
                text = "DO NOT STRAIGHTEN YOUR HAND"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # # Get the correct position to ensure text is within the frame
                # text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                # cv2.putText(img, text, text_position, font, scale, color, thickness)
                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                handBend = True
                if counted == True:
                    handBend = False
                    counted = False
                if handBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            timestamps.append(int(timestamp))

        if len(lmList) < 2 and not camera_warning_shown:
                cv2.putText(img, "Adjust camera position!", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                camera_warning_shown = True   

        timestamps = list(set(timestamps))        
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime
        #cv2.putText(img, f'{fps}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)


        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    #hand_instability(timestamps, img)

#preacherCurls()  

def hammerCurls():
    # r is used because windows automatically puts backslashes when copying path, but we need forward slashes in python
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\hammercurlvid.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    timestamps = []
    timestamp = 0
    camera_warning_shown = False
    ui_width = 170
    backBend = False
    start_time = time.time()
    lineColor = (0,255,0)
    counted = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        # img = cv2.imread("test.jpg")
        # img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) < 2 and not camera_warning_shown:
                cv2.putText(img, "Adjust camera position!", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                camera_warning_shown = True
        if len(lmList) != 0:
            #Right Arm 74 Being minimum and 159 being maximum
            angle = detector.findAngle(img, 12, 14, 16)
            hipangle = detector.findAngle(img,12,24,26, lineColor=lineColor )
            #Left Arm
            # angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (60, 160), (100,0))
            bar = np.interp(angle, (60, 160), (100, 650))
            #print(per)

            # Check for the dumbbell curls
            color = (0,255,0)
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
            
            if hipangle > 190:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                text = "KEEP YOUR BACK STRAIGHT"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            timestamps.append(int(timestamp))
            

        timestamps = list(set(timestamps))
        color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green        
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime
        #cv2.putText(img, f'{fps}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)
    #hip_instability(timestamps, img)
       
#hammerCurls()
def bencHTricepDips():
    # r is used because windows automatically puts backslashes when copying path, but we need forward slashes in python
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\benchdips.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    camera_warning_shown = False
    ui_width = 170
    lineColor = (0,255,0)
    kneeBend = False
    counted = False
    start_time = time.time()
    timestamps = []
    timestamp = None
    while True:
        success, img = cap.read()
        if not success:
            break
        # img = cv2.imread("test.jpg")
        # img.resize((1280, 720))
        # img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) < 2 and not camera_warning_shown:
                cv2.putText(img, "Adjust camera position!", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                camera_warning_shown = True
        

        if len(lmList) != 0:
            #Right Arm 74 Being minimum and 159 being maximum
            angle = detector.findAngle(img, 12, 14, 16,)
            kneeAngle = detector.findAngle(img, 24, 26, 28, lineColor=lineColor)
            #Left Arm
            # angle = detector.findAngle(img, 11, 13, 15)
            
            if kneeAngle > 200 or kneeAngle < 170:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                text = "KEEP YOUR LEGS STRAIGHT"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                kneeBend = True
                if counted == True:
                    kneeBend = False
                    counted = False
                if kneeBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            timestamps.append(int(timestamp))

            per = np.interp(angle, (90, 159), (100,0))
            bar = np.interp(angle, (90, 159), (100, 650))
            #print(per)

            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green
            # Check for the dumbbell curls
            color = (0,255,0)
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

        

        timestamps = list(set(timestamps))
        
        color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green        
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime
        #cv2.putText(img, f'{fps}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas) 
        cv2.waitKey(1)
    #legStraight(timestamps,img)
#bencHTricepDips()

def deadlift():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\backbendCorrect.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    timestamps = []
    backBend = False
    start_time = time.time()
    counted = False
    timestamp = 0
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        #img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) != 0:
            # Calculate backbend angle
            backbend_angle = calculate_backbend_angle(lmList)
            angle = detector.findAngle(img, 11, 23, 25)
            kneeangle = detector.findAngle(img, 23,25,27, lineColor=lineColor)

            per = np.interp(angle, (180, 310), (100,0))
            bar = np.interp(angle, (180, 310), (100, 650))
            #print(per)

            # Check for the dumbbell curls
            color = (0,255,0)
            if per == 100:
                color = (0,0,255)
                if dir ==0:
                    count += 1
                    dir = 1
            if per == 0:
                color = (0,255,0)
                if dir == 1:
                    count += 0
                    dir = 0
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if 230 < kneeangle < 300 and per > 60:
                frame_height, frame_width = img.shape[:2]
                text = "LIFT YOUR BODY ALTOGETHER"
                #position = (250, 500)
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                # Get the correct position to ensure text is within the frame

                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                camera_warning_shown = True
                shoulder_backBend = True
                if shoulder_counted == True:
                    shoulder_backBend = False
                    shoulder_counted = False
                if shoulder_backBend == True:
                    shoudler_end_time = time.time()
                    shoulder_timestamp = shoudler_end_time - start_time
                    shoulder_counted = True

            if backbend_angle > 55:
                frame_height, frame_width = img.shape[:2]
                text = "KEEP YOUR BACK STRAIGHT"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2
                UI_Color = (0, 0, 0)
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            # You can add further actions here when backbend is detected

            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            timestamps.append(int(timestamp))
                
            
                        #Right Arm 74 Being minimum and 159 being maximum
            
            #Left Arm
            # angle = detector.findAngle(img, 11, 13, 15)
            

            if len(lmList) < 2 and not camera_warning_shown:
                text = "BODY NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)
                camera_warning_shown = True

        timestamps = list(set(timestamps))

        color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime



        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)


# deadlift()

def romaniandeadlifts():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\backbendCorrect.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    timestamps = []
    backBend = False
    start_time = time.time()
    counted = False
    timestamp = 0
    ui_width = 170
    lineColor = (0,255,0)
    shoulder_counted = False
    shoulder_backBend = False
    shoulder_timestamp = 0
    shoulder_timestamps = []
    while True:
        success, img = cap.read()
        if not success:
            break
        #img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) != 0:
            # Calculate backbend angle
            backbend_angle = calculate_backbend_angle(lmList)
            angle = detector.findAngle(img, 11, 23, 25)
            kneeangle = detector.findAngle(img, 23,25,27, lineColor=lineColor)

            per = np.interp(angle, (180, 310), (100,0))
            bar = np.interp(angle, (180, 310), (100, 650))
            #print(per)

            # Check for the dumbbell curls
            color = (0,255,0)
            if per == 100:
                color = (0,0,255)
                if dir ==0:
                    count += 1
                    dir = 1
            if per == 0:
                color = (0,255,0)
                if dir == 1:
                    count += 0
                    dir = 0
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees

            if backbend_angle > 55:
                frame_height, frame_width = img.shape[:2]
                text = "KEEP YOUR BACK STRAIGHT"
                position = (250, 250)
                lineColor = (0,0,255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2
                UI_Color = (0, 0, 0)
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            # You can add further actions here when backbend is detected
            if kneeangle < 100: 
                frame_height, frame_width = img.shape[:2]
                text = "KNEE BENDING TOO MUCH"
                #position = (250, 500)
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                lineColor = (0,0,255)
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                # Get the correct position to ensure text is within the frame

                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                camera_warning_shown = True
                shoulder_backBend = True
                if shoulder_counted == True:
                    shoulder_backBend = False
                    shoulder_counted = False
                if shoulder_backBend == True:
                    shoudler_end_time = time.time()
                    shoulder_timestamp = shoudler_end_time - start_time
                    shoulder_counted = True

            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                lineColor = (0,255,0)
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            timestamps.append(int(timestamp))
                
            
                        #Right Arm 74 Being minimum and 159 being maximum
            
            #Left Arm
            # angle = detector.findAngle(img, 11, 13, 15)
            

            if len(lmList) < 2 and not camera_warning_shown:
                text = "BODY NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)
                camera_warning_shown = True

        timestamps = list(set(timestamps))

        color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime



        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

# romaniandeadlifts()

def squat():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\squats.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    timestamps = []
    backBend = False
    start_time = time.time()
    counted = False
    timestamp = 0
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        #img = detector.findPose(img, False)
        if not success:
            break
        img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img_resized, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
                text = "BODY IS NOT VISIBLE"
                position = (250, 250)
                lineColor = (0,0,255)
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                


                    
        if len(lmList) != 0:
            frame_height, frame_width = img.shape[:2]

            # #Right Arm
            angle = detector.findAngle(img, 11, 23, 25,lineColor=lineColor)
            # Calculate backbend angle
            backbend_angle = squat_backbend(lmList)
            
            # Visualize backbend angle on image
            #cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if 90 > backbend_angle > 60:
                frame_height, frame_width = img.shape[:2]
                # Show "backbend" prompt
                text = "YOUR BACK IS BENDING"
                lineColor = (0,0,255)
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            timestamps.append(int(timestamp))
        
            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (190, 285), (0,100))
            bar = np.interp(angle, (190, 285), (650, 100))
            #print(per)
            color = (0,255,0)
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
        
        timestamps = list(set(timestamps))
        color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime

                # You can add further actions here when backbend is detected
        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)
    
    #back_bend(timestamps, img)   


# squat()
# Shoulder Workout
def lateralRaise():
    # r is used because windows automatically puts backslashes when copying path, but we need forward slashes in python
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\lateralraise.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    camera_warning_shown = False
    timestamps = []
    backBend = False
    start_time = time.time()
    counted = False
    timestamp = 0
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img = cv2.imread("test.jpg")
        # img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            #Right Arm 74 Being minimum and 159 being maximum
            angle = detector.findAngle(img, 24,12,16)
            #Left Arm
            # angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (30, 100), (0,100))
            bar = np.interp(angle, (30, 100), (650, 100))
            #print(per)

            # Check for the exercise
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
            

            if len(lmList) < 2 and not camera_warning_shown:
                frame_height, frame_width = img.shape[:2]
                text = "BODY IS NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)
                camera_warning_shown = True
            
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)


                
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime
        #cv2.putText(img, f'{fps}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)


# lateralRaise()


def shoulderpress():
    # r is used because windows automatically puts backslashes when copying path, but we need forward slashes in python
    cap = cv2.VideoCapture(1)

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    camera_warning_shown = False
    timestamps = []
    backBend = False
    start_time = time.time()
    counted = False
    timestamp = 0
    ui_width = 170
    lineColor = (0,255,0)
    color = 0
    bar = 0
    per = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        # img = cv2.imread("test.jpg")
        # img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        frame_height, frame_width = img.shape[:2]

        if len(lmList) != 0:
            #Right Arm 74 Being minimum and 159 being maximum
            angle = detector.findAngle(img, 12,14,16, lineColor=lineColor)
            angle2 = detector.findAngle(img, 11, 13, 15, lineColor=lineColor)
            #Left Arm
            # angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (200, 310), (100, 0))
            bar = np.interp(angle, (200, 310), (100, 650))
            per2 = np.interp(angle2, (40, 160), (0, 100))

            #print(per)

            # Check for the exercise
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
            

            if len(lmList) < 2 and not camera_warning_shown:
                frame_height, frame_width = img.shape[:2]
                text = "BODY IS NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)
                camera_warning_shown = True
            if  per-10 < per2 < per+10:


                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            else:

                frame_height, frame_width = img.shape[:2]
                # Show "backbend" prompt
                text = "RAISE YOUR HANDS TOGETHER"
                lineColor = (0,0,255)
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            
            timestamps.append(timestamp)


        timestamps = list(set(timestamps))
                
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime
        #cv2.putText(img, f'{fps}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

# shoulderpress()

# THIS IS FROM RIGHT SIDE
def frontRaise():
    # r is used because windows automatically puts backslashes when copying path, but we need forward slashes in python
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\frontraise.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    camera_warning_shown = False
    timestamps = []
    backBend = False
    start_time = time.time()
    counted = False
    timestamp = 0
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img = cv2.imread("test.jpg")
        img_resized = resize_image(img, height=720)  # Resize the frame
        img = detector.findPose(img_resized, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            #Right Arm 74 Being minimum and 159 being maximum
            angle = detector.findAngle(img, 24,12,16)
            #Left Arm
            # angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (260, 350), (100,0))
            bar = np.interp(angle, (260, 350), (100, 650))
            #print(per)

            # Check for the exercise
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0

            if len(lmList) < 2 and not camera_warning_shown:
                frame_height, frame_width = img.shape[:2]
                text = "BODY IS NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)
                camera_warning_shown = True

            else:
                if angle < 10:
                    frame_height, frame_width = img.shape[:2]
                    lineColor = (0,255,0)
                    text = "KEEP YOUR HAND INFRONT OF BODY"
                    position = (20, 60)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.6
                    UI_Color = (0, 0, 0)
                    thickness = 2
                    text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                    overlay = img.copy()
                    alpha = 0.7  # Adjust opacity here
                    cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                    # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                    # Get the correct position to ensure text is within the frame
                    
                    cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                    cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                else:
                    frame_height, frame_width = img.shape[:2]
                    lineColor = (0,255,0)
                    text = "PROMPTS WILL SHOW HERE"
                    position = (20, 60)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.6
                    UI_Color = (0, 0, 0)
                    thickness = 2
                    text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                    overlay = img.copy()
                    alpha = 0.7  # Adjust opacity here
                    cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                    # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                    # Get the correct position to ensure text is within the frame
                    
                    cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                    cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            # overlay = img.copy()
            # alpha = 0.5  # Adjust opacity here
            # cv2.rectangle(overlay, (150, 150), (0, 0), color, cv2.FILLED)
            # cv2.rectangle(overlay, (int(bar), 100), (0, 0), color, cv2.FILLED)
            # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # # Draw Rep Count with adjusted opacity
            # overlay = img.copy()
            # alpha = 0.5  # Adjust opacity here
            # cv2.rectangle(overlay, (0, 100), (120, 50), color, cv2.FILLED)
            # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # # Draw Percentage with adjusted opacity
            # overlay = img.copy()
            # alpha = 0.5  # Adjust opacity here
            # cv2.putText(overlay, f'{int(per)} %', (20, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
            # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # # Draw Rep Count with adjusted opacity
            # overlay = img.copy()
            # alpha = 0.5  # Adjust opacity here
            # cv2.putText(overlay, f'{count}', (0, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
            # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

               
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime


        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

# frontRaise()
#Forearms Workout

# THIS IS FROM RIGHT SIDE
def wristCurls():
    # r is used because windows automatically puts backslashes when copying path, but we need forward slashes in python
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\wristcurrl.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    camera_warning_shown = False
    timestamps = []
    backBend = False
    start_time = time.time()
    counted = False
    timestamp = 0
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img = cv2.imread("test.jpg")
        # img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            #Right Arm 74 Being minimum and 159 being maximum
            angle = detector.findAngle(img, 14,16,20, lineColor=lineColor)
            #Left Arm
            # angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (145, 159), (100,0))
            bar = np.interp(angle, (145, 159), (100, 650))
            #print(per)

            # Check for the exercise
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0

            if len(lmList) < 2 and not camera_warning_shown:
                frame_height, frame_width = img.shape[:2]
                text = "BODY IS NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)
                camera_warning_shown = True

            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime


        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)
            # overlay = img.copy()
            # alpha = 0.5  # Adjust opacity here
            # cv2.rectangle(overlay, (150, 150), (0, 0), color, cv2.FILLED)
            # cv2.rectangle(overlay, (int(bar), 100), (0, 0), color, cv2.FILLED)
            # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # # Draw Rep Count with adjusted opacity
            # overlay = img.copy()
            # alpha = 0.5  # Adjust opacity here
            # cv2.rectangle(overlay, (0, 100), (120, 50), color, cv2.FILLED)
            # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # # Draw Percentage with adjusted opacity
            # overlay = img.copy()
            # alpha = 0.5  # Adjust opacity here
            # cv2.putText(overlay, f'{int(per)} %', (20, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
            # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # # Draw Rep Count with adjusted opacity
            # overlay = img.copy()
            # alpha = 0.5  # Adjust opacity here
            # cv2.putText(overlay, f'{count}', (0, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
            # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
       
        #cv2.putText(img, f'{fps}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
# wristCurls()

def dumbbellRows():
    # For Importing Video {Devansh}
    start_time = time.time()
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\dumbbell_rows.mp4")
    forwardbend = False
    end_time = 0
    end_time_backward = 0
    backward_bend_elapsed = 0
    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    new_arr = []
    knee_angle_count = 0  # Initialize a variable to count knee angle occurrences
    counted = False
    backwardcounted = False
    backwardbendcount = 0
    backwardbendarr = []
    timestamp = 0
    timestamps=[]
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        img_resized = resize_image(img, width=800)  # Resize the frame
        img_resized = cv2.convertScaleAbs(img_resized)
        img = detector.findPose(img_resized, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) == 0:
            cv2.putText(img, 'Body is not correctly visible', (20, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            

        if len(lmList) != 0:
            angle = detector.findAngle(img, 12, 14, 16)
            knee_angle = detector.findAngle(img, 12, 24, 26)
            distance_elbow_hips = detector.findDistance(img, 14, 12)
            # if distance_elbow_hips > 70:
            #     text = "PLEASE KEEP YOUR FOREARM PARALLEL TO YOUR THIGHS"
            #     position = (250, 250)
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     scale = 1
            #     color = (0, 0, 255)
            #     thickness = 2

            #     # Get the correct position to ensure text is within the frame
            #     text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
            #     cv2.putText(img, text, text_position, font, scale, color, thickness)
            #     camera_warning_shown = True
            
            if knee_angle > 120:
                frame_height, frame_width = img.shape[:2]
                text = "LEAN FORWARD"
                #position = (250, 500)
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                camera_warning_shown = True
                forwardbend = True
                if counted == True:
                    forwardbend = False                
                if forwardbend == True:
                    knee_angle_count += 1  # Increment the count when knee angle is greater than 90
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True


            if knee_angle < 100 and knee_angle > 50:
                counted = False
                backwardcounted = False
        
            elif knee_angle < 50:
                frame_height, frame_width = img.shape[:2]
                text = "LEAN BACKWARD"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                camera_warning_shown = True
                backwardbend = True
                if backwardcounted == True:
                    backwardbend = False                
                if backwardbend == True:
                    backwardbendcount += 1  # Increment the count when knee angle is greater than 90
                    end_time_backward = time.time()
                    backwardcounted = True
                    timestamp = end_time - start_time

            backbend_angle = calculate_backbend_angle(lmList)
            
            if backbend_angle > 70:
                frame_height, frame_width = img.shape[:2]
                text = "YOUR BACK IS BENDING"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                camera_warning_shown = True
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True

            timestamps.append(int(timestamp))

            #Left Arm
            per = np.interp(angle, (105,175 ), (100,0))
            bar = np.interp(angle, (105,175 ), (100, 650))
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green

            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0



        timestamps = list(set(timestamps))        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

        

    # #Ensuring unique filename
    # filename = ensure_unique_file("output.txt")
    
    # # Open a file in write mode
    # with open(filename, "w") as file:
    #     # Write the count of backward leans to the file
    #     file.write("Number of times you leaned backward: " + str(knee_angle_count) + "\n")
    #     # Write the count of forward leans to the file
    #     file.write("Number of times you leaned forward: " + str(backwardbendcount) + "\n")

    #     # Write the times of backward leans to the file
    #     i = 0
    #     while i < len(new_arr):
    #         file.write(f"Time at which you leaned backwards: {new_arr[i]}\n")
    #         i += 1

    #     # Write the times of forward leans to the file
    #     j = 0
    #     while j < len(backwardbendarr):
    #         file.write(f"Time at which you leaned Forward: {backwardbendarr[j]}\n")
    #         j += 1

    #     file.write("Your back was bend at:\n")
    #     for i, timestamp in enumerate(timestamps):
    #         message = f'{time.strftime("%H:%M:%S", time.localtime(timestamp))}\n'
    #         cv2.putText(img, message, (50, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #         file.write(message)
# dumbbellRows()

# FROM LEFT SIDE 
def cableRows():
    # r is used because windows automatically puts backslashes when copying path, but we need forward slashes in python
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\cableRows.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    camera_warning_shown = False
    timestamps = []
    backBend = False
    start_time = time.time()
    counted = False
    timestamp = 0
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img = cv2.imread("test.jpg")
        img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img_resized, False)
        lmList = detector.findPosition(img, False)
        shoulderDistance = detector.findDistance(img, 11, 15)
        if len(lmList) != 0:
            #Right Arm 74 Being minimum and 159 being maximum
            angle = detector.findAngle(img, 11,13,15)
            hipangle = detector.findAngle(img, 25, 23, 11)
            #Left Arm
            # angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (205, 265), (0,100))
            bar = np.interp(angle, (205, 265), (650, 100))
            #print(per)
            if shoulderDistance < 70:
                frame_height, frame_width = img.shape[:2]
                text = "KEEP YOUR HAND CLOSE TO HIPS"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            if hipangle < 80:
                frame_height, frame_width = img.shape[:2]
                text = "DO NOT LEAN THAT FORWARD"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            # Check for the exercise
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))  # Gradual color change from red to green
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0

            if len(lmList) < 2 and not camera_warning_shown:
                frame_height, frame_width = img.shape[:2]
                text = "BODY IS NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)
                camera_warning_shown = True

                
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime
        #cv2.putText(img, f'{fps}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        
        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)
# cableRows()

#LEFT SIDE
def tBarRows():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\tbarrows.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    backBend = False
    start_time = time.time()
    counted = False
    ui_width = 170
    lineColor = (0,255,0)
    #timestamp=0
    #timestamps=[]


    while True:
        success, img = cap.read()
        if not success:
            break
        # img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                text = "BODY IS NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)
                camera_warning_shown = True

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 11, 13, 15)
            knee_angle = detector.findAngle(img, 25, 23,11, lineColor=lineColor)
            if knee_angle > 105 or knee_angle < 65:
                frame_height, frame_width = img.shape[:2]
                text = "PLEASE KEEP YOUR HIPS AND BACK STABLE"
                lineColor = (0,0,255)
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            # Calculate backbend angle
            backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if backbend_angle > 55:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "YOUR BACK IS BENDING"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                camera_warning_shown = True
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            timestamps.append(int(timestamp))

            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (200, 270), (0,100))
            bar = np.interp(angle, (200, 270), (650, 100))
            #print(per)
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100))) 
            # Check for the dumbbell curls
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
        timestamps = list(set(timestamps))
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

# tBarRows()

#LEFT SIDE
def pushup():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\pushups.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    pTime = 0
    #timestamps = []
    backBend = False
    start_time = time.time()
    counted = False
    #timestamp = 0
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img_resized = resize_image(img ,width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
            cv2.putText(img, 'Body is not correctly visible', (20, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 11, 13, 15)
            knee_angle = detector.findAngle(img, 25, 23,11)
            if knee_angle > 180 or knee_angle < 160:
                text = "PLEASE KEEP YOUR BODY STRAIGHT"
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            
            # Calculate backbend angle
            backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if backbend_angle > 60:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "YOUR BACK IS BENDING"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                camera_warning_shown = True
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (198, 270), (0,100))
            bar = np.interp(angle, (198, 270), (650, 100))
            #print(per)
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            # Check for the dumbbell curls
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
     
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)
# pushup()

# FROM LEFT SIDE
def chestdips():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\chest dips.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    start_time = time.time()
    counted = False
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img_resized = resize_image(img ,width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
                text = "BODY NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 11, 13, 15 )
            knee_angle = detector.findAngle(img, 25, 23,11, lineColor=lineColor)
            if knee_angle > 190 or knee_angle < 65:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                text = "PLEASE KEEP YOUR HIPS AND BACK STABLE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (198, 270), (100,0))
            bar = np.interp(angle, (198, 270), (100, 450))
            #print(per)
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            # Check for the dumbbell curls
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0

            # Calculate backbend angle
            backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if backbend_angle > 190:
                frame_height, frame_width = img.shape[:2]
                # Show "backbend" prompt
                text = "YOUR BACK IS BENDING"
                lineColor = (0,0,255)
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
                # You can add further actions here when backbend is detected
            
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)
# chestdips()


def benchpress():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\benchpress.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    start_time = time.time()
    counted = False
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img_resized = resize_image(img ,width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
                text = "BODY NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 12, 14, 16 )
            # knee_angle = detector.findAngle(img, 25, 23,11, lineColor=lineColor)
            backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if backbend_angle > 55:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                camera_warning_shown = True
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "MAINTAIN ARCH IN BACK"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (68, 160), (0,100))
            bar = np.interp(angle, (68, 160), (650, 100))
            #print(per)
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            # Check for the dumbbell curls
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
            
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

# benchpress()


def bulgarianSplits():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\bulgarianhuman.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    start_time = time.time()
    counted = False
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img_resized = resize_image(img ,width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
                text = "BODY NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 24, 26, 28, lineColor=lineColor )
            anotherfoot = detector.findAngle(img, 23, 25, 27)
            # knee_angle = detector.findAngle(img, 25, 23,11, lineColor=lineColor)
            backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if angle < 50:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "MOVE A BIT FORWARD"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            if backbend_angle > 55:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "KEEP YOUR BACK STRAIGHT"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                camera_warning_shown = True
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (70, 135), (0,100))
            bar = np.interp(angle, (70, 135), (650, 100))
            #print(per)
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            # Check for the dumbbell curls
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
            
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

# bulgarianSplits()

def lunges():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\lunges.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    start_time = time.time()
    counted = False
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img_resized = resize_image(img ,width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
                text = "BODY NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 24, 26, 28, lineColor=lineColor )
            another_foot = detector.findAngle(img, 23, 25, 27, lineColor=lineColor)
            # knee_angle = detector.findAngle(img, 25, 23,11, lineColor=lineColor)
            backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if angle < 50:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "MOVE A BIT FORWARD"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            if backbend_angle > 55:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "KEEP YOUR BACK STRAIGHT"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                camera_warning_shown = True
                backBend = True
                if counted == True:
                    backBend = False
                    counted = False
                if backBend == True:
                    end_time = time.time()
                    timestamp = end_time - start_time
                    counted = True
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (75, 155), (0,100))
            bar = np.interp(angle, (75, 155), (650, 100))
            #print(per)
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            # Check for the dumbbell curls
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
            
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)
# lunges()

#CORE EXERRCISES 
def planks():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\planks.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    start_time = time.time()
    counted = False
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img_resized = resize_image(img ,width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
                text = "BODY NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 11, 23, 25, lineColor=lineColor )
            # knee_angle = detector.findAngle(img, 25, 23,11, lineColor=lineColor)
            backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if 170 > angle or angle > 203:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "STRAIGHTEN YOUR BODY"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (140, 185), (0,100))
            bar = np.interp(angle, (140, 185), (650, 100))
            #print(per)
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            # Check for the dumbbell curls
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
            
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), (0,0,255), cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'FORM METER', (frame_width+20, 45), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

# planks()

def legRaises():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\legraises.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    start_time = time.time()
    counted = False
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img_resized = resize_image(img ,width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
                text = "BODY NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 11, 23, 25, lineColor=lineColor )
            # knee_angle = detector.findAngle(img, 25, 23,11, lineColor=lineColor)
            backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if angle < 185:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "DO NOT LOOSE TENSION"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (190, 280), (0,100))
            bar = np.interp(angle, (190, 280), (650, 100))
            #print(per)
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            # Check for the dumbbell curls
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
            
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'reps: {int(count)}', (frame_width+20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)
# legRaises()

def crunches():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\crunches.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    start_time = time.time()
    counted = False
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img_resized = resize_image(img ,width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
                text = "BODY NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 25, 23, 11 )
            knee_angle = detector.findAngle(img, 27, 25, 23, lineColor=lineColor)
            # backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if knee_angle < 190:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "BEND YOUR KNEES"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (120, 145), (100,0))
            bar = np.interp(angle, (120, 145), (100, 650))
            #print(per)
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            # Check for the dumbbell curls
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
            
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'reps: {int(count)}', (frame_width+20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

# crunches()

def reverseCrunches():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\reverse_crunches.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    start_time = time.time()
    counted = False
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img_resized = resize_image(img ,width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
                text = "BODY NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 25, 23, 11 )
            knee_angle = detector.findAngle(img, 27, 25, 23, lineColor=lineColor)
            # backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if angle > 165:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "ELEVATE YOUR LEGS"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(knee_angle, (200, 300), (0,100))
            bar = np.interp(knee_angle, (200, 300), (650, 100))
            #print(per)
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            # Check for the dumbbell curls
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
            
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'reps: {int(count)}', (frame_width+20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

# reverseCrunches()


def mountainClimbers():
    # For Importing Video {Devansh}
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\mountainClimbers.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    start_time = time.time()
    counted = False
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        if not success:
            break
        # img_resized = resize_image(img ,width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) == 0:
                text = "BODY NOT VISIBLE"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)

        if len(lmList) != 0:
            # #Right Arm
            angle = detector.findAngle(img, 15,11,23, lineColor=lineColor )
            knee_angle = detector.findAngle(img, 11, 23, 25)
            # backbend_angle = calculate_backbend_angle(lmList)
            
            # Visualize backbend angle on image
            # cv2.putText(img, f'Backbend Angle: {backbend_angle}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Check if backbend angle is more than 60 degrees
            if angle < 250 or angle > 290:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,0,255)
                # Show "backbend" prompt
                text = "KEEP YOUR UPPER BODY STRAIGHT"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.5  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

            #Left Arm
            # detector.findAngle(img, 11, 13, 15)
            per = np.interp(knee_angle, (215, 300), (0,100))
            bar = np.interp(knee_angle, (215, 300), (650, 100))
            #print(per)
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
            # Check for the dumbbell curls
            if per == 100:
                color = (0,255,0)
                if dir ==0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0,0,255)
                if dir == 1:
                    count += 0.5
                    dir = 0
            
        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'reps: {int(count)}', (frame_width+20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

# mountainClimbers()

#FROM LEFT SIDE
def tricepropepushdown():
    # r is used because windows automatically puts backslashes when copying path, but we need forward slashes in python
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\tricepropepushdown.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    camera_warning_shown = False
    start_time = time.time()
    counted = False
    ui_width = 170
    lineColor = (0,255,0)

    while True:
        success, img = cap.read()
        # img = cv2.imread("test.jpg")
        if not success:
            break
        # img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) < 2 and not camera_warning_shown:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "BODY NOT"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
                camera_warning_shown = True
        if len(lmList) != 0:
            #Right Arm 74 Being minimum and 159 being maximum
            angle = detector.findAngle(img, 11,13,15)
            shoulder_angle = detector.findAngle(img, 13, 11, 23)
            #Left Arm
            # angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (190, 270), (100,0))
            bar = np.interp(angle, (190, 270), (650, 100))
            #print(per)

            # Check for the exercise
            color = (0, int(255 * (per / 100)), int(255 * (1 - per / 100)))
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
            if shoulder_angle > 350:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "RAISE YOUR SHOULDER"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            if shoulder_angle < 310:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "LOWER YOUR SHOULDER"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)
            else:
                frame_height, frame_width = img.shape[:2]
                lineColor = (0,255,0)
                text = "PROMPTS WILL SHOW HERE"
                position = (20, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                UI_Color = (0, 0, 0)
                thickness = 2
                text_position = put_text_in_frame(img, text, position, font, scale, (0,0,255), thickness)
                overlay = img.copy()
                alpha = 0.7  # Adjust opacity here
                cv2.rectangle(overlay, (frame_width, 100), (0, 0), UI_Color, cv2.FILLED)
                # cv2.rectangle(overlay, (int(bar), 100), (0, 0), (255,255,255), cv2.FILLED)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                # Get the correct position to ensure text is within the frame
                
                cv2.rectangle(overlay, (frame_width,100), (0,0), UI_Color, cv2.FILLED)
                cv2.putText(img, text, text_position, font, scale, (255,255,255), thickness)

        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime
        #cv2.putText(img, f'{fps}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        
        canvas = np.zeros((frame_height, frame_width + ui_width, 3), dtype=np.uint8)

        # Overlay background image onto canvas
        canvas[:, :frame_width] = img
        # Draw the bar on the right side of the canvas
        cv2.rectangle(canvas, (frame_width, 100), (frame_width + 170, 0), color, cv2.FILLED)
        cv2.rectangle(canvas, (frame_width, int(bar)), (frame_width + 170, frame_height), color, cv2.FILLED)
        cv2.putText(canvas, f'{int(per)}%', (frame_width + 60, int(bar) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Draw Rep Count on the right side of the canvas
        cv2.putText(canvas, f'Reps: {int(count)}', (frame_width + 20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


        cv2.imshow("Image with UI", canvas)
        cv2.waitKey(1)

#tricepropepushdown()

def triceppushdown():
    # r is used because windows automatically puts backslashes when copying path, but we need forward slashes in python
    cap = cv2.VideoCapture(r"Exercise_Vids\Exercises\TricepPushdown.mp4")

    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    camera_warning_shown = False

    while True:
        success, img = cap.read()
        # img = cv2.imread("test.jpg")
        img_resized = resize_image(img, width=800)  # Resize the frame
        img = detector.findPose(img_resized, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            #Right Arm 74 Being minimum and 159 being maximum
            angle = detector.findAngle(img, 11,13,15)
            shoulder_angle = detector.findAngle(img, 13, 11, 23)
            #Left Arm
            # angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (200, 270), (100,0))
            bar = np.interp(angle, (200, 270), (650, 100))
            #print(per)

            # Check for the exercise
            color = (0,255,0)
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

            if shoulder_angle > 350:
                text = "RAISE YOUR SHOULDER"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)
            if shoulder_angle < 310:
                text = "LOWER YOUR SHOULDER"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)

            if len(lmList) < 2 and not camera_warning_shown:
                text = "ADJUST CAMERA POSITION"
                position = (250, 250)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                color = (0, 0, 255)
                thickness = 2

                # Get the correct position to ensure text is within the frame
                text_position = put_text_in_frame(img, text, position, font, scale, color, thickness)
                cv2.putText(img, text, text_position, font, scale, color, thickness)
                camera_warning_shown = True

            overlay = img.copy()
            alpha = 0.5  # Adjust opacity here
            cv2.rectangle(overlay, (150, 150), (0, 0), color, cv2.FILLED)
            cv2.rectangle(overlay, (int(bar), 100), (0, 0), color, cv2.FILLED)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # Draw Rep Count with adjusted opacity
            overlay = img.copy()
            alpha = 0.5  # Adjust opacity here
            cv2.rectangle(overlay, (0, 100), (120, 50), color, cv2.FILLED)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # Draw Percentage with adjusted opacity
            overlay = img.copy()
            alpha = 0.5  # Adjust opacity here
            cv2.putText(overlay, f'{int(per)} %', (20, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # Draw Rep Count with adjusted opacity
            overlay = img.copy()
            alpha = 0.5  # Adjust opacity here
            cv2.putText(overlay, f'{count}', (0, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime
        #cv2.putText(img, f'{fps}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
# triceppushdown()

