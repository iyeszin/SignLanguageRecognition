import argparse
from model import Model
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import copy
import cv2 as cv
from keras import models
from utils import helper

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
        print(self.cp_img.size)
        self.evaluate_model(self.cp_img)

    def getImgCallback(self, imgpath):
        self.img = cv.imread(imgpath)
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        print(self.img.size)
        self.img = cv.resize(self.img, [28, 28])
        self.img = cv.flip(self.img, 1)
        print(self.img.size)
        self.cp_img = copy.deepcopy(self.img)
        # print(self.cp_img)
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

if __name__ == '__main__':
    inference_img = Inference(img_path)