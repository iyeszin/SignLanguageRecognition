import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras
from keras.preprocessing.image import ImageDataGenerator
from model import Model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from utils import helper


# parameters (convert into arguments)
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
# INPUT_SHAPE = (28, 28, 1)
NUMBER_OF_CLASSESS = 24

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--modeltype", type=str, default = "res18", help="to specific which model architecture to train, ie, cnn, res18, res34")
ap.add_argument("-e", "--epochs", type=int, default = 5, help="epochs num")
ap.add_argument("-bs", "--batchsize", type=int, default = 128, help="batchsize")
ap.add_argument("-lr", "--learningrate", type=float, default = 0.001, help="initial learning rate")
ap.add_argument("-train", "--traindf", type=str, default = "/home/iyeszin/Documents/sl_recognizer/dataset/sign_mnist_train/sign_mnist_train.csv", help="path to train dataset")
ap.add_argument("-test", "--testdf", type=str, default = "/home/iyeszin/Documents/sl_recognizer/dataset/sign_mnist_test/sign_mnist_test.csv", help="path to test dataset")
ap.add_argument("-name", "--modelname", type=str, default = "./sl-cnn-weight.hdf5", help="name to specific which model to save")
args = vars(ap.parse_args())

model_type = args["modeltype"]
nb_epochs = args["epochs"]
bs = args["batchsize"]
LR = args["learningrate"]
train = args["traindf"]
test = args["testdf"]
mn = args["modelname"]

optimizer = Adam(learning_rate=LR)
cce_loss = keras.losses.CategoricalCrossentropy(from_logits=False)

cb_list = [
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            verbose=1,
            monitor="val_loss",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            f"best_acc_{mn}.h5",
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
        ),
    ]


print("[INFO] Building the model...")
if model_type == "res18":
    model = Model.ResNet18(INPUT_SHAPE, NUMBER_OF_CLASSESS)
elif model_type == "res34":
    model = Model.ResNet34(INPUT_SHAPE, NUMBER_OF_CLASSESS)
else:
    model = Model.cnn(INPUT_SHAPE, NUMBER_OF_CLASSESS)

print(model.summary())

print("[INFO] Loading data generator...")
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

datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)

datagen.fit(train_x)

print("[INFO] Network info...")
# showing information to input into network



print("[INFO] Training the network...")
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(
    datagen.flow(train_x,y_train, batch_size = bs),
    epochs=nb_epochs,
    validation_data=(test_x,y_test),
    validation_steps=total_test//bs,
    steps_per_epoch=total_train//bs,
    callbacks=cb_list)

model.save('saved_models/model.h5')

# show learning curves
plt.plot(history.history['accuracy'], 'g-o', label='train')
plt.plot(history.history['val_accuracy'], 'r-o', label='test')
plt.title('Training & validation accuracy', pad=-80)
plt.legend()
plt.savefig('saved_models/'+mn+'_accuracy.png')
# plt.show()


plt.plot(history.history['loss'], 'g-o', label='train')
plt.plot(history.history['val_loss'], 'r-o', label='test')
plt.title('Training & validation loss', pad=-80)
plt.legend()
plt.savefig('saved_models/'+mn+'_loss.png')
# plt.show()