  
  
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
  
_about ann here..._
  
####  Layers
  
  
**Flatten**  
_about layer here..._
  
**Dense Layer**  
_about layer here..._
  
####  Model Validations
  
_model configurations and data here..._
  
###  Convolutional Neural Network (CNN)
  
_about cnn here..._
  
####  Layers
  
**Convolutional Layer**  
_about layer here... (kernels)_
  
**Max Pooling Layer**  
_about layer here... (kernels)_
  
**Dropout (Regularization)**
_about layer here..._
  
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
  
_about face detector..._  
[FaceDetector.py](src/FaceDetectorpp.py )
  
###  FaceRecognizer (.py)
  
_about face recognizer..._  
[FaceRecognizer.py](src/FaceRecognizer.py )
  
###  DataGenerator (.py)
  
_about data generator..._  
[DataGenerator.py](src/DataGenerator.py )
  
###  Utilities (.py)
  
_about utilities..._  
[Utilities.py](src/Utilities.py )
  
###  App (.py)
  
_about flask app..._  
[FaceDetector.py](src/app.py )
  
###  models (directory)
  
_about the models..._
  