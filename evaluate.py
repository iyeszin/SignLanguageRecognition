import argparse
from model import Model
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from utils import helper
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


INPUT_SHAPE = (28, 28, 1)
NUMBER_OF_CLASSESS = 24


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-train", "--traindf", type=str, default = "/home/iyeszin/Documents/sl_recognizer/dataset/sign_mnist_train.csv", help="path to train dataset")
ap.add_argument("-test", "--testdf", type=str, default = "/home/iyeszin/Documents/sl_recognizer/dataset/sign_mnist_test.csv", help="path to test dataset")
ap.add_argument("-m", "--modeltype", type=str, default = "res34", help="to specific which model architecture to train, ie, cnn, res18, res34")
ap.add_argument("-w", "--weight", type=str, default = "weight/res34-weight.hdf5", help="path to specific which model architecture to train")
args = vars(ap.parse_args())

train = args["traindf"]
test = args["testdf"]
model_type = args["modeltype"]
weight = args["weight"]


print("[INFO] Processing data...")
train_df = pd.read_csv(train)
test_df = pd.read_csv(test)

total_train = train_df.shape[0]
total_test = test_df.shape[0]


y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)
train_x = helper.preprocess_image(train_df.values)
test_x = helper.preprocess_image(test_df.values)
x_train = train_df.values
x_test = test_df.values


LR = 1e-4 
optimizer = Adam(learning_rate=LR)


def evaluating_accuracy():
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

    print('------------------------------------------------------------------------')
    print(f'Performance of Test Data')
    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top_k_categorical_accuracy", dtype=None)
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_k_categorical_accuracy", dtype=None)
    pred_y_test = model.predict(test_x)
    top1.update_state(y_true=y_test,y_pred=pred_y_test)
    print('top1 Acc')
    print(top1.result())
    top5.update_state(y_true=y_test, y_pred=pred_y_test)
    print('top5 Acc')
    print(top5.result())

    return 0



if __name__ == '__main__':
    evaluating_accuracy()