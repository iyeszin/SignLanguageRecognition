import argparse
from model import Model
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import copy
import cv2 as cv

class_names = ["A","B","C","D","E","F","G","H","I","K","L",'M','N','O','P','Q','R','S','T','U','V','W','X','Y']

# param
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1
N_IMAGES = 1

INPUT_SHAPE = (28, 28, 1)
NUMBER_OF_CLASSESS = 24

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-img", "--imagepath", type=str, default = "/home/iyeszin/Documents/sl_recognizer/a.jpg" , help="path to dataset")
ap.add_argument("-m", "--modeltype", type=str, default = "res34", help="to specific which model architecture to train, ie, cnn, res18, res34")
ap.add_argument("-w", "--weight", type=str, default = "weight/res34-weight.hdf5", help="path to specific which model architecture to train")
args = vars(ap.parse_args())

img_path = args["imagepath"]
model_type = args["modeltype"]
weight = args["weight"]

# print(model_type)

class Inference():
   
    def __init__(self):
        super().__init__()
        self.IMAGE_WIDTH = 28
        self.IMAGE_HEIGHT = 28
        self.IMAGE_CHANNELS = 1
        self.N_IMAGES = 1
        self.INPUT_SHAPE = (28, 28, 1)
        self.NUMBER_OF_CLASSESS = 24
        self.model_type = model_type
        # self.cp_img = []
        self.pixels = []
        # self.model = self.model

        self.cp_img = self.getImgCallback()
        print(self.cp_img.size)
        self.evaluate_model()

    def getImgCallback(self):
        self.img = cv.imread(img_path)
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        print(self.img.size)
        self.img = cv.resize(self.img, [28, 28])
        self.img = cv.flip(self.img, 1)
        print(self.img.size)
        self.cp_img = copy.deepcopy(self.img)
        # print(self.cp_img)
        return self.cp_img


        # self.img = Image.open(img_path)
        # print(self.img.size)
        # self.img = self.img.resize((28,28))
        # self.img = self.img.convert('L')
        # self.img = self.img.rotate(-90)
        # self.cp_img = copy.deepcopy(self.img)
        # self.pixels = np.array(self.img.getdata()).reshape((1,28,28))/255

        # return pixels
        # plt.imshow(pixels.reshape((28,28)),cmap='gray')


    def evaluate_model(self):
        if self.model_type == "res18":
            self.model = Model.ResNet18(INPUT_SHAPE, NUMBER_OF_CLASSESS)
        elif self.model_type == "res34":
            self.model = Model.ResNet34(INPUT_SHAPE, NUMBER_OF_CLASSESS)
        elif self.model_type == "cnn":
            self.model = Model.cnn(INPUT_SHAPE, NUMBER_OF_CLASSESS)
        else:
            print("no model is load")
    
        #------ load data
        # print("[INFO] Reading image...")
        pixels = np.array(self.img).reshape((1,28,28))/255
        print(pixels.size)
        # plt.imshow(pixels.reshape((28,28)),cmap='gray')

        # ------ inferenece
        if self.model_type == "cnn":
            pixels = pixels.reshape(N_IMAGES, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
        my_images_preds = self.model.predict(pixels)
        my_images_preds = np.argmax(my_images_preds)
        print(my_images_preds)
        print("Predicted letter is: "+ class_names[my_images_preds])

# def main(args=None):
#     inference_img = Inference()

if __name__ == '__main__':
    inference_img = Inference()