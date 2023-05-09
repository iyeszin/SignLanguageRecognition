import argparse
from model import Model
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import copy
import cv2 as cv
from keras import models
from utils import helper
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
mp_model = mp_hands.Hands(
    static_image_mode=True, # only static images
    max_num_hands=2, # max 2 hands detection
    min_detection_confidence=0.5) # detection confidence

output_dir = "output/"
class_names = ["A","B","C","D","E","F","G","H","I","K","L",'M','N','O','P','Q','R','S','T','U','V','W','X','Y']

# param
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1
N_IMAGES = 1

INPUT_SHAPE = (28, 28, 1)
NUMBER_OF_CLASSESS = 24
MODEL_NAME = "/home/iyeszin/Documents/sl_recognizer/saved_models_res18/model.h5"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-img", "--imagepath", type=str, default = "/home/iyeszin/Documents/sl_recognizer/Images/20230220_084429.jpg" , help="path to dataset")
ap.add_argument("-m", "--modeltype", type=str, default = "res34", help="to specific which model architecture to train, ie, cnn, res18, res34")
args = vars(ap.parse_args())

img_path = args["imagepath"]
model_type = args["modeltype"]

class Inference():
   
    def __init__(self, imgpath):
        super().__init__()
        self.IMAGE_WIDTH = 28
        self.IMAGE_HEIGHT = 28
        self.IMAGE_CHANNELS = 1
        self.N_IMAGES = 1
        self.INPUT_SHAPE = (28, 28, 1)
        self.NUMBER_OF_CLASSESS = 24
        self.model_type = model_type
        self.pixels = []

        self.cp_img = self.getImgCallback(imgpath)
        self.evaluate_model(self.cp_img)

    def get_bbox_coordinates(self, handLadmark, image_shape):
        """ 
        Get bounding box coordinates for a hand landmark.
        Args:
            handLadmark: A HandLandmark object.
            image_shape: A tuple of the form (height, width).
        Returns:
            A tuple of the form (xmin, ymin, xmax, ymax).
        """
        all_x, all_y = [], [] # store all x and y points in list
        for hnd in mp_hands.HandLandmark:
            all_x.append(int(handLadmark.landmark[hnd].x * image_shape[1])) # multiply x by image width
            all_y.append(int(handLadmark.landmark[hnd].y * image_shape[0])) # multiply y by image height

        return min(all_x), min(all_y), max(all_x), max(all_y) # return as (xmin, ymin, xmax, ymax)

    def getImgCallback(self, imgpath):
        self.img = cv.imread(imgpath)

        results = mp_model.process(cv.cvtColor(self.img, cv.COLOR_BGR2RGB))

        # add bounding box 
        image_height, image_width, c = self.img.shape # get image shape

        # aspect ratio to crop the image to fit into 640 x 640
        aspect_ratio = image_height/image_width
        # iterate on all detected hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            xmin, ymin, xmax, ymax = self.get_bbox_coordinates(hand_landmarks, (image_height, image_width))
            # uncomment the following for landmark drawing
            # mp_drawing.draw_landmarks(
            #     self.img, # image to draw
            #     hand_landmarks, # model output
            #     mp_hands.HAND_CONNECTIONS, # hand connections
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style())
            
        xmin -= int(40 * (2 + aspect_ratio))
        ymin -= int(40 * (2 + aspect_ratio))
        xmax += int(40 * (2 + aspect_ratio))
        ymax += int(40 * (2 + aspect_ratio))
        cropped_image = self.img[ymin:ymax, xmin:xmax]

        # flip and write output image to disk
        cv.imwrite(f"{output_dir}/crop.jpg", cv.flip(cropped_image, 1))

        gray = cv.cvtColor(cropped_image, cv.COLOR_RGB2GRAY)
        
        # Create a kernel for morphological operations
        kernel = np.ones((5,5),np.uint8)
        # Perform erosion to remove small objects
        img_erosion = cv.erode(gray, kernel, iterations=1)

        # Perform dilation to fill in gaps
        img_dilation = cv.dilate(img_erosion, kernel, iterations=1)

        resized_image= cv.resize(img_dilation, [self.IMAGE_WIDTH, self.IMAGE_HEIGHT])
        resized_image = cv.flip(resized_image, 1)

        self.cp_img = copy.deepcopy(resized_image)

        return self.cp_img


    def evaluate_model(self, img):

        self.model = models.load_model(MODEL_NAME)
    
        #------ load data
        print("[INFO] Reading image...")
        pixels = helper.preprocess_image(img)

        # ------ inferenece
        my_images_preds = self.model.predict(pixels)
        label = np.argmax(my_images_preds)
        print(my_images_preds)
        print("Predicted letter is: "+ class_names[label])
        if np.max(my_images_preds) > 0.9:
            label = np.argmax(my_images_preds)
            print("0.9 Predicted letter is: "+ class_names[label])
        else:
            print("ohh nooooo")
        

if __name__ == '__main__':
    inference_img = Inference(img_path)