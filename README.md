  
  
#  Exam Project 2020 | Face Recognizer
  
By **Pernille Lørup & Stephan Djurhuus**  
Institute **CPHBusiness**  
  
Education **Software Development**  
Elective **Artificial Intelligence**  
  
[Link to GitHub Repository](https://github.com/BA-Software-Development-Ed/ai-exam-project )
  
###  Objective
  
The objective of this task is to enable you to demonstrate the knowledge of artificial intelligence and machine learning acquired during the elective AI course.  
The task is to create a machine learning based solution to a real life problem.
  
Full exam details can be found in [exam-task.md](exam-task.md ).
  
###  Prerequisite
  
**[Anaconda Environment](https://anaconda.org/ )**
  
additional python packages
```bash
Flask           1.1.2       # pip install Flask 
tensorflow      2.1.0       # pip install tensorflow
opencv-python   4.2.0.34    # pip install opencv-python
```
  
###  Execution
  
To start the flask application run the following script and go to http://localhost:5000/.
  
**Run Server**  
_bash_
```bash
python src/App.py
```
  
###  Notebooks
  
The notebooks is located here [src/notebooks](src/notebooks ).
  
___
##  The Content
  
  
- [The Theory](#the-theory )
  - [Introduction](#introduction )
  - [Face Detection](#face-detection )
    - [Cascade Classification](#cascade-classification )
      - [Classifier Validations](#classifier-validations )
  - [Data Processing & Augmentation](#data-processing-augmentation )
  - [General Neural Network](#general-neural-network )
    - [Loss Functions](#loss-functions )
    - [Adam Optimizer](#adam-optimizer )
  - [Artificial Neural Network (ANN)](#artificial-neural-network-ann )
    - [Layers](#layers )
    - [Model Validations](#model-validations )
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn )
    - [Layers](#layers-1 )
    - [Model Validations](#model-validations-1 )
  - [Conclusion](#conclusion )
- [The Source](#the-source )
  - [Project Structure](#project-structure )
  - [FaceDetector (.py)](#facedetector-py )
  - [FaceRecognizer (.py)](#facerecognizer-py )
  - [DataGenerator (.py)](#datagenerator-py )
  - [Utilities (.py)](#utilities-py )
  - [App (.py)](#app-py )
  - [models (directory)](#models-directory )
  
___
##  The Theory
  
  
###  Introduction
  
We decided to create a face recognition application, using `TensorFlow`, `openCV` and `Flask` as the main components. 
  
The models are based on each family member in `Family1` from the [datasets](src/data/PersonGroup )
  
![face recognition](assets/face-recognition.png )
  
The core functionality of the system is to isolate faces in images and make a recognition based on the highest predicted label.
  
The project also includes a web application to interact with the model as a client. This application uses the best of our models to detect and recognize the faces in the posted images.  
  
###  Face Detection
  
  
####  Cascade Classification
  
  
We are using a _Haar feature-based cascade classifier_ from `openCV`, to detect faces bounding boxes on the images. The reason for this is to isolate the face as much possible to optimize the recognizer model.
  
Haar feature-based cascade classifiers are pretrained classifiers made for a specific purpose, in our case detecting faces on images. Like many other intelligent agents it has been trough a supervised learning with great amount of images with and without faces. The features in the classifier is similar to the convolutional kernel in a CNN model.
  
#####  Classifier Validations
  
  
![classifier detections](assets/classifier-detections.png )
  
**haarcascade_frontalcatface_alt**  
_validation here..._
  
**haarcascade_frontalcatface_alt2**  
_validation here..._
  
**haarcascade_frontalcatface_default**  
_validation here..._
  
[reference, docs.opencv.org](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html )
[resource, github.com/opencv](https://github.com/opencv/opencv/tree/master/data/haarcascades )
  
###  Data Processing & Augmentation
  
  
![data generation](assets/data-generation.png )
  
We are using our custom class [FaceDetector.py](src/FaceDetector.py ) to crop the face from each image. 
  
Subsequently we are using the custom class [DataGenerator.py](src/DataGenerator.py ) to generate augmented images. This class uses `ImageDataGenerator` from Tensor Flow, to manipulate images with given arguments.
  
  
```python
ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10,
    horizontal_flip=True,
    fill_mode='reflect',  # constant, nearest, reflect, wrap
)
```
  
As the last step of our data processing we split the dataset into training and testing datasets.
  
[ImageDataGenerator, tensorflow.org](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator )
  
###  General Neural Network
  
  
####  Loss Functions
  
_about CategoricalCrossentropy..._
_about SparseCategoricalCrossentropy..._
  
####  Adam Optimizer
  
The `Adam` optimizer is the recommended optimizer for general purposes because the default configuration parameters work well with most problems. The optimizer is a gradient decent method and creates good and fast results.
  
###  Artificial Neural Network (ANN)
  
An Artificial Neural Network is a collection of connected nodes/neurons which can pass a value from one neuron to another. It consists of three types of layers; input, hidden and output. These layer connections forms the network architecture. 
  
####  Layers
  
  
**Flatten**  
Flatten layer takes an <img src="https://latex.codecogs.com/gif.latex?n"/> dimensional array and makes it one dimensional and are easier for the model to process. 
  
**Dense Layer**  
This layer creates a vector of <img src="https://latex.codecogs.com/gif.latex?n"/> neurons that uses the output from the previous layer. 
  
####  Model Validations
  
_model configurations and data here..._
  
###  Convolutional Neural Network (CNN)
  
A Convolutional Neural Network is a type of ANN but has one or more layers consisting of convolutional units. 
  
A convolutional layer creates n outputs generated by the kernels. Kernels add all the field's pixel values with the coefficient from the pixel's location in the kernel. These outputs are new images based on the input, but assigned with the results of the kernels.
  
![convolutional image](https://miro.medium.com/max/790/1*1VJDP6qDY9-ExTuQVEOlVg.gif )
  
####  Layers
  
  
**Max Pooling Layer**  
A Max Pooling layer creates an output generated by the kernels. Kernels takes the highest pixel value from the pixels located in the kernel. This creates a new image based on the input, but assigned with the results of the kernel.
  
**Dropout (Regularization)**
The Dropout layer excludes a random <img src="https://latex.codecogs.com/gif.latex?n"/> percentage of the dataset for each epoch. This reduces the case of over-fitting, but can cause under-fitting if the dropout is too high.
  
####  Model Validations
  
_model configurations and data here..._
  
###  Conclusion
  
_conclusion here..._
  
with these analysis we found that the kids looks more like their mother than their father.
  
![image of family detection]( )
___
##  The Source
  
  
###  Project Structure
  
  
```bash
# jupyter notebooks for demonstration
/notebooks
├─ FaceDetector.ipynb     # face detection
└─ FaceRecognizer.ipynb   # face recognition
  
# flask files
App.py      # main file
/templates   # html templates
/static      # javascript, styling and assets
  
# custom classes for flask application
FaceDetector.py     # notebook as class
FaceRecognizer.py   # notebook as class
DataGenerator.py   # notebook as class
Utilities.py        # classes for encoding, plotting ect.
  
# models and classifiers
/cascade-classifiers   # facial cascade classifiers
/models                # saved tensorflow models
  
# image collections
/data # dataset from microsoft
```
  
###  FaceDetector (.py)
  
[FaceDetector.py](src/FaceDetectorpp.py )
  
The class contains methods to crop, resize, mark and get face details and uses the _Haar feature-based cascade classifier_ to detect faces in images. 
  
###  FaceRecognizer (.py)
  
[FaceRecognizer.py](src/FaceRecognizer.py )
  
The class contains methods to train and test the model, save and load models, get model summary and recognize faces from previous detections.
  
###  DataGenerator (.py)
  
[DataGenerator.py](src/DataGenerator.py )
  
The class contains methods to generate augmented datasets from cropped images, merge and shuffle datasets.
  
###  Utilities (.py)
  
[Utilities.py](src/Utilities.py )
  
This module contains a `Base64` class that contains methods to decode base64 images.
This module also contains a `Displayer` class that contains methods to plot images, confusion matrix, accuracy history and mark detections with predictions.
  
###  App (.py)
  
[FaceDetector.py](src/App.py )
  
This is the flask server main file, containing the routes and endpoints for the web application.
  
###  models (directory)
  
This directory contains the pretrained models with their corresponding accuracy history.
  
The models are saved in this format: `TYPE_model_Xof4.h5`  
The history are saved in this format: `TYPE_model_Xof4_hist.csv`
  