## Sign Language Recognition

This project aims to detect isolated Sign Language Characters by building three models, cnn, resnet18, and resnet34. The models were trained using the sign language mnist dataset, which is similar in format to the classic MNIST dataset. The training data consists of 27,455 cases, and the test data has 7172 cases. Each case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (except for J and Z due to gesture motions). Each image is 28x28 pixels with grayscale values between 0-255.
Installation

To install the project, clone the repository and navigate to the project directory:

``` bash
git clone https://github.com/iyeszin/SignLanguageRecognition.git
cd sign-language-recognition
```

# Usage

To train the model, run the following command:

```python
python train_model.py -m res34 -name "./res34-weight.hdf5"
```
This command will train the resnet34 model and save the weights to a file named "res34-weight.hdf5". You can replace "res34" with "cnn" or "res18" to train the cnn or resnet18 models, respectively.

To perform inference on a test image, run the following command:

```python
python inference.py -img "Images/20230220_084429.jpg" -m res34 -w "weight/res34-weight.hdf5"
```
This command will perform inference on the test image located at "Images/20230220_084429.jpg" using the resnet34 model with the weights loaded from the "res34-weight.hdf5" file. You can replace "res34" with "cnn" or "res18" to use the cnn or resnet18 models, respectively.

# Acknowledgments

The sign language mnist dataset was used in this project. Thanks to the authors of the dataset for making it available.