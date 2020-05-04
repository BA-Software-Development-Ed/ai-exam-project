  
  
#  Exam Project 2020 | Face Recognizer
  
By **Pernille Lørup & Stephan Djurhuus**  
Institute **CPHBusiness**  
  
Education **Software Development**  
Elective **Artificial Intelligence**  
  
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
python src/app.py
```
  
___
##  The Content
  
  
  - [Prerequisite](#prerequisite )
  - [Execution](#execution )
- [The Theory](#the-theory )
  - [Face Detection](#face-detection )
  - [Face Recognition](#face-recognition )
  - [Neural Network](#neural-network )
    - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn )
- [The Source](#the-source )
  - [Project Structure](#project-structure )
  
___
##  The Theory
  
  
###  Face Detection
  
  
###  Face Recognition
  
  
###  Neural Network
  
  
####  Convolutional Neural Network (CNN)
  
  
___
##  The Source
  
  
###  Project Structure
  
  
```bash
# jupyter notebooks for demonstration
notebooks
  FaceDetector.py     # face detection
  FaceRecognizer.py   # face recognition
  
# flask files
app.py      # main file
templates   # html templates
static      # javascript, styling and assets
storage     # stored images from webcams
  
# custom classes for flask application
FaceDetector.py     # notebook as class
FaceRecognizer.py   # notebook as class
Utilities.py        # classes for encoding, plotting ect.
  
# models and classifiers
cascade-classifiers   # facial cascade classifiers
models                # saved tensorflow models
```
  
  