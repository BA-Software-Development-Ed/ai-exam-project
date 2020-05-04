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

___
## The Content {ignore=true}
[TOC]

___
## The Theory

### Face Detection

### Face Recognition

### Neural Network

#### Convolutional Neural Network (CNN)

___
## The Source

### Project Structure

```bash
# jupyter notebooks for demonstration
notebooks
  FaceDetector.ipynb     # face detection
  FaceRecognizer.ipynb   # face recognition

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

