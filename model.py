import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPooling2D , Flatten , Dropout , BatchNormalization, Activation

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1

L2_WEIGHT_DECAY = 2e-4
INPUT_SHAPE = (28, 28, 1)
NUMBER_OF_CLASSESS = 24

# Common parameters to tune the model
embedding_size = 128 # 32, 128, 512
dropout_rate = 0.5 # 0.0, 0.5, 0.9
nb_epochs = 20
bs = 128


class Model:
    @staticmethod
    def identity_block(x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        # Layer 2
        x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])     
        x = tf.keras.layers.Activation('relu')(x)
        return x

    @staticmethod
    def convolutional_block(x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        # Layer 2
        x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        # Processing Residue with conv(1,1)
        x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])     
        x = tf.keras.layers.Activation('relu')(x)
        return x

    @staticmethod
    def ResNet18(shape = INPUT_SHAPE, classes = NUMBER_OF_CLASSESS):
        # Step 1 (Setup Input Layer)
        x_input = tf.keras.layers.Input(shape)
        x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
        # Step 2 (Initial Conv layer along with maxPool)
        x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        # Define size of sub-blocks and initial filter size
        block_layers = [2, 2, 2, 2]
        filter_size = 64
        # Step 3 Add the Resnet Blocks
        for i in range(4):
            if i == 0:
                # For sub-block 1 Residual/Convolutional block not needed
                for j in range(block_layers[i]):
                    x = Model.identity_block(x, filter_size)
            else:
                # One Residual/Convolutional Block followed by Identity blocks
                # The filter size will go on increasing by a factor of 2
                filter_size = filter_size*2
                x = Model.convolutional_block(x, filter_size)
                for j in range(block_layers[i] - 1):
                    x = Model.identity_block(x, filter_size)
        # Step 4 End Dense Network
        x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation = 'relu')(x)
        x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
        model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet18")
        return model

    @staticmethod
    def ResNet34(shape = INPUT_SHAPE, classes = NUMBER_OF_CLASSESS):
        # Step 1 (Setup Input Layer)
        x_input = tf.keras.layers.Input(shape)
        x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
        # Step 2 (Initial Conv layer along with maxPool)
        x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        # Define size of sub-blocks and initial filter size
        block_layers = [3, 4, 6, 3]
        filter_size = 64
        # Step 3 Add the Resnet Blocks
        for i in range(4):
            if i == 0:
                # For sub-block 1 Residual/Convolutional block not needed
                for j in range(block_layers[i]):
                    x = Model.identity_block(x, filter_size)
            else:
                # One Residual/Convolutional Block followed by Identity blocks
                # The filter size will go on increasing by a factor of 2
                filter_size = filter_size*2
                x = Model.convolutional_block(x, filter_size)
                for j in range(block_layers[i] - 1):
                    x = Model.identity_block(x, filter_size)
        # Step 4 End Dense Network
        x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation = 'relu')(x)
        x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
        model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
        return model
    
    @staticmethod
    def cnn(shape = INPUT_SHAPE, classes = NUMBER_OF_CLASSESS):
        # define model
        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(embedding_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        
        return model

if __name__ == '__main__':
    model = Model.ResNet34(INPUT_SHAPE, NUMBER_OF_CLASSESS)
    model.summary()