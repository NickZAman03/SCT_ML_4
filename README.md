# Hand Gesture Recognition with CNN
This repository contains a Jupyter Notebook for building a Hand Gesture Recognition Model using a Convolutional Neural Network (CNN). The model leverages the Leap GestRecog dataset to train and classify various hand gestures.

**Features**
Dataset Integration: Automatically downloads the dataset from Kaggle.
Image Augmentation: Uses ImageDataGenerator for enhancing model generalization.
CNN Architecture: Employs TensorFlow/Keras for building a robust model.
Performance Visualization: Includes training history plots (accuracy and loss).
**File Overview**
Hand_Gesture.ipynb: The main notebook containing code for downloading the dataset, preprocessing, model training, and evaluation.
**Dataset**
The notebook uses the Leap GestRecog dataset, which contains gesture images. The dataset is fetched using kagglehub.

Dataset Source: LeapGestRecog on Kaggle

**Preprocessing Steps**
Resize images to a uniform dimension.
Normalize pixel values for improved model performance.
Augment data using techniques like flipping, rotation, and zoom.
Model Architecture
The model is a CNN built with:

Convolutional Layers: For feature extraction.
Pooling Layers: For down-sampling.
Fully Connected Layers: For gesture classification.
Dependencies
The following Python libraries are required:

* kagglehub
* numpy
* tensorflow
* matplotlib
* os
**Install dependencies using:**

bash

`pip install kagglehub numpy tensorflow matplotlib`
**Usage Instructions**
* Clone the repository:

bash

`git clone https://github.com/your-username/your-repo-name.git`
`cd your-repo-name`
* Run the notebook: Open the notebook Hand_Gesture.ipynb in Jupyter or JupyterLab and execute the cells step by step.

* Dataset Download: Ensure you have a Kaggle API token saved in your environment to access the dataset.

* Train the Model: Follow the notebook steps to preprocess data, train the model, and evaluate its accuracy.

**Results**
The model achieves significant accuracy in gesture recognition, and the training progress is visualized using plots.

**Future Improvements**
Extend support for additional gesture datasets.
Optimize the model for real-time recognition.
Build a standalone application for gesture recognition.


**Acknowledgments**
Kaggle for providing the Leap GestRecog dataset.
TensorFlow/Keras for the model-building tools.
