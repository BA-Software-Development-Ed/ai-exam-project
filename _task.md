# Implementations
The applications overall functionality is to follow theses protocols.

**Face Detection Protocol**
1. Capture image from webcam
2. Send image to server for processing
3. Detect face from image
6. Apply face box image
7. Send new image back to client

**Face Recognition Protocol**
1. Capture image from webcam
2. Send image to server for processing
3. Detect face from image
4. Crop and reshape face from image
5. Use recognizer on cropped image
6. Apply face box and name on original image
7. Send new image back to client

**Create Profile Protocol**
1. Capture multiple images from webcam
2. Send images to server for processing
3. Detect face from images
4. Crop and reshape face from images
5. Train model with faces from images 
5. Capture image from webcam with _Face Recognition Protocols_

**Does these models don't fulfill the criteria**  
Implementation of `ANN` vs `CNN` would be a solution.

_fast commit_
```bash
git add . && git commit -m "fast commit" && git push
```
___
## Task
### 1. Face Detection
- [x] Face Detector Notebook 
- [x] Face Detector Class, based on notebook 
- [x] Utility Classes, based on notebook util functions

### 2. Face Recognition
- [x] Face Recognizer Notebook 
- [x] Face Recognizer Class, based on notebook  
- [x] Utility Classes, based on notebook util functions

### 3.1 Web Application
- [x] Implement Face Detector Class
- [x] Implement Face Recognizer Class

### 3.2 Web Application Redesign
- [x] index.html => detect image & recognize
- [x] detect.html => post images, response with face box image
- [x] recognize.html => post images, response with face box image, and names - maybe percentage

### 4. Optimizing
- [ ] Add `Unknown` class as result, maybe by percentage
- [x] Combine ANN & CNN in single notebook
- [x] Make several ANN & CNN models, and compare
- [x] Add markdown content to notebooks
- [x] Save diagrams as images for the `_docs.md`
- [ ] Finnish `_docs.md`
- [x] Short function comments in `.py` files
- [ ] Describe & Fix Classifiers
- [x] Get Pernille up and Running
- [ ] Add Accuracy on images
- [x] Add shadow on conclusion image
- [x] Fix app.py to App.py