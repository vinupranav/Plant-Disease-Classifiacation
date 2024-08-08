# Potato Disease Classification

![Banner Image](https://jpinfotech.org/wp-content/uploads/2023/01/JPPY2210-Plant-Disease-Detection-and-Classification.jpg) <!-- Replace with your banner image URL -->

## Overview

This project aims to classify potato plant diseases using a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras. The dataset used contains images of potato plants categorized into three classes: Early Blight, Late Blight, and Healthy. The model is designed to identify the disease from images with high accuracy.

## Data

The dataset for this project is sourced from the PlantVillage dataset, which contains labeled images of potato plants. The data is divided into three main categories:
- **Potato___Early_blight**
- **Potato___Late_blight**
- **Potato___healthy**

## Data Preprocessing

1. **Loading Data**:
   The dataset is loaded using TensorFlow's `image_dataset_from_directory` function, which handles the images in batches and resizes them to a uniform size of 256x256 pixels.

2. **Dataset Partitioning**:
   The dataset is split into training, validation, and test sets using an 80-10-10 ratio:
   - Training Set: 80%
   - Validation Set: 10%
   - Test Set: 10%

3. **Caching and Prefetching**:
   To improve performance, caching and prefetching are applied to the training, validation, and test datasets.

4. **Data Augmentation**:
   Data augmentation techniques, such as random flipping and rotation, are used to enhance the model's generalization ability by artificially expanding the training dataset.

## Model Architecture

The CNN model consists of several layers:

- **Input Layer**:
  - Resizing and normalization of images (256x256 pixels, pixel values scaled to [0,1]).

- **Convolutional Layers**:
  - Several Conv2D layers with increasing depth and MaxPooling2D layers to extract and downsample features from images.

- **Fully Connected Layers**:
  - Dense layers to interpret the features and produce a final classification.

- **Output Layer**:
  - Softmax activation to classify the input into one of the three categories.

The model architecture is as follows:

- Conv2D + MaxPooling2D (multiple layers)
- Flatten
- Dense (64 units)
- Dense (3 units, softmax activation)

## Training

The model is trained for 50 epochs using the Adam optimizer and SparseCategoricalCrossentropy loss function. The training process involves monitoring accuracy on the validation set to prevent overfitting and ensuring the model generalizes well to unseen data.

## Results

Training and validation accuracy are recorded during the training process. The final model's performance is evaluated on the test set to determine its classification accuracy.

## Usage

To use the model for classification:
1. Load the model.
2. Preprocess the input image to match the required input format.
3. Use the model's `predict` method to classify the image.

