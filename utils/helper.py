# function helpers

import numpy as np
import matplotlib.pyplot as plt
import string
from collections import deque
import cv2 as cv


letters = dict(zip(list(range(0,26)),string.ascii_lowercase))

unique_labels = [ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24]
# unique_letters = [letters[x] for x in y_train]
# unique_letters.sort()

def preprocess_image(x):
    
    """
    we know that the pixcel values lies between 0-255 but it is obsearved that models performs exceptionally well if we scale pixel values
    between 0-1"""
    x = x/255 # normalize the data
    x = x.reshape(-1,28,28,1) # convert it into 28 x 28 gray scaled image
    
    return x



def show_images(images,labels):
    """
    take images array as input and display the image 
    """
    fig,ax = plt.subplots(2,5)
    fig.set_size_inches(10, 6)
    k = 0
    for i in range(2):
        for j in range(5):
            ax[i,j].imshow(images[k] , cmap='gray')
            # ax[i,j].set_title(str(unique_labels[np.argmax(images[k])]))
            k = k+1;
    plt.tight_layout()


def predictions_to_labels(pred):
    labels =[]
    for p in pred:
        labels.append(unique_labels[np.argmax(p)])
    return labels


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded