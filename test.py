# import cv2
# import mediapipe as mp

# output_dir = "output/"
# image_path = "/home/iyeszin/Documents/sl_recognizer/a.jpg"

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands

# # For static images:
# mp_model = mp_hands.Hands(
#     static_image_mode=True, # only static images
#     max_num_hands=2, # max 2 hands detection
#     min_detection_confidence=0.5) # detection confidence

# # we are not using tracking confidence as static_image_mode is true.

# image = cv2.imread(image_path)
# # now we flip image and convert to rgb image and input to model
# image = cv2.flip(image, 1)

# results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# # Get handedness
# print(results.multi_handedness)
# # cv2.imshow('img',image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# image_height, image_width, c = image.shape # get image shape
# # iterate on all detected hand landmarks
# for hand_landmarks in results.multi_hand_landmarks:
#       # we can get points using mp_hands
#       print(f'Ring finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height})'
#       )

# # for hand_landmarks in results.multi_hand_landmarks:
# #     mp_drawing.draw_landmarks(
# #         image, # image to draw
# #         hand_landmarks, # model output
# #         mp_hands.HAND_CONNECTIONS, # hand connections
# #         mp_drawing_styles.get_default_hand_landmarks_style(),
# #         mp_drawing_styles.get_default_hand_connections_style())
    


# def get_bbox_coordinates(handLadmark, image_shape):
#     """ 
#     Get bounding box coordinates for a hand landmark.
#     Args:
#         handLadmark: A HandLandmark object.
#         image_shape: A tuple of the form (height, width).
#     Returns:
#         A tuple of the form (xmin, ymin, xmax, ymax).
#     """
#     all_x, all_y = [], [] # store all x and y points in list
#     for hnd in mp_hands.HandLandmark:
#         all_x.append(int(handLadmark.landmark[hnd].x * image_shape[1])) # multiply x by image width
#         all_y.append(int(handLadmark.landmark[hnd].y * image_shape[0])) # multiply y by image height

#     return min(all_x), min(all_y), max(all_x), max(all_y) # return as (xmin, ymin, xmax, ymax)
    
# xmin, ymin, xmax, ymax = get_bbox_coordinates(hand_landmarks, (image_height, image_width, c))
# # draw bounding box on image
# # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

# print(xmin, ymin, xmax, ymax)
# xmin -= 40
# ymin -= 40
# xmax += 40
# ymax += 40
# cropped_image = image[ymin:ymax, xmin:xmax]

# # flip and write output image to disk
# # cv2.imwrite(f"{output_dir}/{image_path.split('/')[-1]}", cv2.flip(image, 1))
# # cv2.imwrite(f"{output_dir}/crop.jpg", cv2.flip(cropped_image, 1))


# def crop():
#     # reading image
#     image = cv2.imread(image_path)

#     # converting to gray scale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # applying canny edge detection
#     edged = cv2.Canny(gray, 10, 250)

#     # finding contours
#     (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     idx = 0
#     for c in cnts:
#         x, y, w, h = xmin, ymin, xmax, ymax
#         if w > 50 and h > 50:
#             idx += 1
#             new_img = image[y + 1 :h, x: w]
#             # cropping images
#             cv2.imwrite(f"{output_dir}/test.jpg", cv2.flip(new_img, 1))
#     print('Objects Cropped Successfully!')

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# train_df = pd.read_csv('/home/iyeszin/Documents/sl_recognizer/dataset/sign_mnist_train/sign_mnist_train.csv')

# def to_image(array, label = True):
#     # Reshape an array into an image format
#     array = np.array(array)
#     start_idx = 1 if label else 0
#     return array[start_idx:].reshape(28,28).astype(float)
        
# # Display one image
# print(train_df.iloc[0])
# img = to_image(train_df.iloc[0])
# # plt.imshow(img, cmap = 'gray')
# # plt.show()

# import cv2
# import mediapipe
 
# drawingModule = mediapipe.solutions.drawing_utils
# handsModule = mediapipe.solutions.hands
 
# capture = cv2.VideoCapture(0)
 
# with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
 
#     while (True):
 
#         ret, frame = capture.read()
#         results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 
#         if results.multi_hand_landmarks != None:
#             for handLandmarks in results.multi_hand_landmarks:
#                 drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
 
#         cv2.imshow('Test hand', frame)
 
#         if cv2.waitKey(1) == 27:
#             break
 
# cv2.destroyAllWindows()
# capture.release()

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import imutils

# global variables
bg = None

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    cnts, _ = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    start_recording = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width = 700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            cv2.destroyAllWindows()
            camera.release()
            break
        
        if keypress == ord("s"):
            start_recording = True

def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))

def showStatistics(predictedClass, confidence):

    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "one"
    elif predictedClass == 1:
        className = "two"
    elif predictedClass == 2:
        className = "three"
    elif predictedClass == 3:
        className = "four"
    elif predictedClass == 4:
        className = "five"
    elif predictedClass == 5:
        className = "fist"
    elif predictedClass == 6:
        className = "L"
    elif predictedClass == 7:
        className = "swing"
    elif predictedClass == 8:
        className = "palm"
    elif predictedClass == 9:
        className = "rock on"
    elif predictedClass == 10:
        className = "blank"

    cv2.putText(textImage,"Pedicted Class : " + className, 
    (30, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    (30, 100), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)
    cv2.imshow("Statistics", textImage)


main()