---
export_on_save:
  markdown: true
markdown:
  path: README.md
  ignore_from_front_matter: true
---

# Exam Project 2020 | Face Recognizer {ignore=true}
By **Pernille Lørup & Stephan Djurhuus**  
Institute **CPHBusiness**  

Education **Software Development**  
Elective **Artificial Intelligence**  

[Link to GitHub Repository](https://github.com/BA-Software-Development-Ed/ai-exam-project)

### Objective {ignore=true}
The objective of this task is to enable you to demonstrate the knowledge of artificial intelligence and machine learning acquired during the elective AI course.  
The task is to create a machine learning based solution to a real life problem.

Full exam details can be found in [exam-task.md](exam-task.md).

### Prerequisite {ignore=true}
**[Anaconda Environment](https://anaconda.org/)**

additional python packages
```bash
Flask           1.1.2       # pip install Flask 
tensorflow      2.1.0       # pip install tensorflow
opencv-python   4.2.0.34    # pip install opencv-python
```

### Execution {ignore=true}
To start the flask application run the following script and go to http://localhost:5000/.

**Run Server**
_bash_
```bash
python src/app.py
```

### Notebooks {ignore=true}
The notebooks is located here [src/notebooks](src/notebooks).

___
## The Content {ignore=true}
[TOC]

___
## The Theory

### Introduction
We decided to create a face recognition application, using `TensorFlow`, `openCV` and `Flask` as the main components. 

![face recognition](assets/face-recognition.png)

_some info about the project..._

### Face Detection

We are using a _Haar feature-based cascade classifier_ from `openCV`, to detect faces bounding boxes on the images. The reason for this is to isolate the face as much possible to optimize the recognizer model.

Haar feature-based cascade classifiers are pretrained classifiers made for a specific purpose, in our case detecting faces on images. Like many other intelligent agents it has been trough a supervised learning with great amount of images with and without faces. The features in the classifier is similar to the convolutional kernel in a CNN model.

#### Classifier Validations

![classifier detections](assets/classifier-detections.png)

**haarcascade_frontalcatface_alt**
_image here..._

**haarcascade_frontalcatface_alt2**
_image here..._

**haarcascade_frontalcatface_default**
_image here..._


#### Cascade Classification
[reference, docs.opencv.org](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)
[resource, github.com/opencv](https://github.com/opencv/opencv/tree/master/data/haarcascades)

### Data Processing

![data generation](assets/data-generation.png)

_about data generator here..._

### Face Recognition

### Artificial Neural Network (ANN)
_about ann here..._

#### Layers

**Flatten**
_about layer here..._

**Dense Layer**
_about layer here..._

#### Activation Functions
_about activation functions here..._

#### Model Validations
_model configurations and data here..._

### Convolutional Neural Network (CNN)
_about cnn here..._

#### Layers
**Convolutional Layer**
_about layer here... (kernels)_

**Max Pooling Layer**
_about layer here... (kernels)_

#### Activation Functions
_about activation functions here..._

#### Model Validations
_model configurations and data here..._

### Conclusion
_conclusion here..._
___
## The Source

### Project Structure

```bash
# jupyter notebooks for demonstration
/notebooks
├─ FaceDetector.ipynb             # face detection
├─ FaceRecognizer.ipynb           # cnn face recognition
└─ ArtificialNeuralNetwork.ipynb  # ann face recognition

# flask files
app.py      # main file
/templates   # html templates
/static      # javascript, styling and assets
/storage     # stored images from webcams

# custom classes for flask application
FaceDetector.py     # notebook as class
FaceRecognizer.py   # notebook as class
Utilities.py        # classes for encoding, plotting ect.

# models and classifiers
/cascade-classifiers   # facial cascade classifiers
/models                # saved tensorflow models

# image collections
/data # dataset from microsoft
```

