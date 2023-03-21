import argparse
from model import Model
from utils import helper
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time
import timeit

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

# # Start the timer
# start_time = time.time()

print(model_type)

def evaluate_model():

    # ------ init model
    if model_type == "res18":
        model = Model.ResNet18(INPUT_SHAPE, NUMBER_OF_CLASSESS)
    elif model_type == "res34":
        model = Model.ResNet34(INPUT_SHAPE, NUMBER_OF_CLASSESS)
    elif model_type == "cnn":
        model = Model.cnn(INPUT_SHAPE, NUMBER_OF_CLASSESS)
        # model = CNN.build_CNNmodel(input_image, NUMBER_OF_CLASSESS)
    else:
        print("no model is load")

    model.load_weights(weight)

    # ------ load data
    # # THIS IS INPUT AS RANDOM FROM TEST DATA
    # test_df = pd.read_csv("/home/iyeszin/Documents/sl_recognizer/dataset/sign_mnist_test.csv")
    # del test_df['label']
    # print(test_df.shape)
    # test_x = helper.preprocess_image(test_df.values)

    # preds = model.predict(test_x)

    # n=8
    # plt.imshow(test_x[n].reshape(28,28),cmap="gray") 
    # plt.grid(False) 
    # print("Predicted letter is:",class_names[np.argmax(preds[n])])


    # THIS IS INPUT AS PHOTO
    print("[INFO] Reading image...")
    image = Image.open(img_path)
    image = image.resize((28,28))
    image = image.convert('L')
    image = image.rotate(-90)

    pixels = np.array(image.getdata()).reshape((1,28,28))/255
    plt.imshow(pixels.reshape((28,28)),cmap='gray')

    # ------ inferenece
    if model_type == "cnn":
        pixels = pixels.reshape(N_IMAGES, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    my_images_preds = model.predict(pixels)
    my_images_preds = np.argmax(my_images_preds)
    print(my_images_preds)
    print("Predicted letter is: "+ class_names[my_images_preds])

# # End the timer
# end_time = time.time()

# # Calculate the elapsed time
# elapsed_time = end_time - start_time

# # Print the elapsed time
# print("Time taken: {:.2f} seconds".format(elapsed_time))

# ------ execution time
excution_time = timeit.timeit(evaluate_model, number=100) # time taken to execute the statement a specified number of times. 
print("Execution time: ", excution_time)
