# Implementations

## Tasks
### Send base64 encoded image back to browser like so
```python
np_array = np.array(...)
base64_string = base64.b64encode(np_array)
```

```html
<img src=`data:image/jpeg;base64,${base64 String}`>
```

**Face Detection Open CV**
https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
  
**Cascade Classifiers**
https://github.com/opencv/opencv/tree/master/data/haarcascades

## Issue

### Plotting error
thread error causes server crash

