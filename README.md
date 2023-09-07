## Sign Language Recognition

This project aims to detect isolated Sign Language Characters by building three models, cnn, resnet18, and resnet34. The models were trained using the sign language mnist dataset, which is similar in format to the classic MNIST dataset. The training data consists of 27,455 cases, and the test data has 7172 cases. Each case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (except for J and Z due to gesture motions). Each image is 28x28 pixels with grayscale values between 0-255.

![ASL fingering](/dataset/american_sign_language.PNG)

# Installation

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
python inference.py -img "Images/20230220_084429.jpg"
```
This command will perform inference on the test image located at "Images/20230220_084429.jpg" using the model loaded from the file.

# Acknowledgments

The sign language mnist dataset was used in this project. Thanks to the authors of the dataset for making it available.

## Notes
When it comes to classical feature extraction techniques from raw pixel data, there are several common approaches that can be used. These techniques aim to extract meaningful information from the pixel values to represent the images in a more structured and informative manner. Here are a few examples:

1. Histogram of Oriented Gradients (HOG): HOG is a popular technique for object detection and image classification. It calculates the distribution of gradient orientations within an image, capturing the local edge and shape information. HOG features can be computed by dividing the image into small cells, calculating gradient orientations within each cell, and creating histograms of these orientations. The histograms are then normalized to capture the overall structure of the image.

2. Scale-Invariant Feature Transform (SIFT): SIFT is a widely used technique for extracting keypoint-based features. It identifies distinctive local features by detecting keypoints in an image and computing descriptors around these keypoints. SIFT features are invariant to scale, rotation, and affine transformations, making them robust to variations in image appearance.

3. Local Binary Patterns (LBP): LBP is a texture descriptor that characterizes the local texture patterns in an image. It encodes the relationships between a pixel and its neighboring pixels by comparing their intensity values. LBP features are computed by dividing the image into small regions, comparing the intensity values of each pixel with its neighbors, and constructing binary patterns. Histograms of these patterns are then computed to capture the texture information.

4. Haralick Texture Features: Haralick texture features describe the texture properties of an image by analyzing the spatial relationships between pixels. These features are derived from the gray-level co-occurrence matrix (GLCM), which captures the occurrences of pairs of pixel intensities at different pixel distances and directions. Haralick features, such as entropy, contrast, and homogeneity, quantify various aspects of the texture patterns in the image.

5. Color Histograms: Color histograms capture the distribution of colors within an image. They represent the frequency of different color values or color channels in the image. Color histograms can be computed separately for each color channel (e.g., RGB, HSV) or in a joint color space representation (e.g., RGB combined into a single histogram).

6. Principal Component Analysis (PCA): PCA is a dimensionality reduction technique that can be applied to the raw pixel data. It transforms the high-dimensional pixel values into a lower-dimensional representation by finding the orthogonal axes that capture the maximum variance in the data. The resulting principal components can be used as features to represent the images.

These are just a few examples of classical feature extraction techniques from raw pixel data. Each technique captures different aspects of the image information, such as gradients, textures, colors, or shapes. The choice of technique depends on the specific requirements of your application and the type of information you want to extract from the images.
