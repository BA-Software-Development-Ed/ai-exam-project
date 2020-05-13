# Implementations {ignore=true}
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

## Content {ignore=true}
[TOC]

___
## Task
### 1. Face Detection
- [x] Face Detector Notebook 
- [x] Face Detector Class, based on notebook 
- [x] Utility Classes, based on notebook util functions

### 2. Face Recognition
- [ ] Face Recognizer Notebook 
- [ ] Face Recognizer Class, based on notebook  
- [ ] Utility Classes, based on notebook util functions

### 3. Web Application
- [ ] Implement Face Detector Class
- [ ] Implement Face Recognizer Class

___
## Sketches
**Send base64 encoded image back to browser like so**

_python_
```python
np_array = np.array(...)
base64_string = base64.b64encode(np_array)
```

_html_
```html
<img src=`data:image/jpeg;base64,${base64 String}`>
```

___
## Issue

### Server Crash on Plotting
Plotting images with open cv from flask application causes thread error and server crash.

**Solution** _I think this is caused because flask isn't allowed to open any new windows. It wouldn't make sense to open windows from the server application anyway, so that is probably a good thing :)._


