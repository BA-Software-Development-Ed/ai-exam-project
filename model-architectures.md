# Model Architectures

## Artificial Neural Network (ANN)

### 2 of 4
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_1 (Flatten)          (None, 10000)             0         
_________________________________________________________________
dense_2 (Dense)              (None, 500)               5000500   
_________________________________________________________________
dense_3 (Dense)              (None, 250)               125250    
_________________________________________________________________
dense_4 (Dense)              (None, 4)                 1004      
=================================================================
Total params: 5,126,754
Trainable params: 5,126,754
Non-trainable params: 0
_________________________________________________________________
```

### 3 of 4
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_6 (Flatten)          (None, 10000)             0         
_________________________________________________________________
dense_17 (Dense)             (None, 500)               5000500   
_________________________________________________________________
dense_18 (Dense)             (None, 250)               125250    
_________________________________________________________________
dense_19 (Dense)             (None, 4)                 1004      
=================================================================
Total params: 5,126,754
Trainable params: 5,126,754
Non-trainable params: 0
_________________________________________________________________
```

## Convolutional Neural Network (CNN)

### 2 of 4
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 98, 98, 25)        700       
_________________________________________________________________
dropout (Dropout)            (None, 98, 98, 25)        0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 49, 49, 25)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 47, 47, 50)        11300     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 23, 23, 50)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 21, 21, 50)        22550     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 10, 10, 50)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 50)          22550     
_________________________________________________________________
flatten (Flatten)            (None, 3200)              0         
_________________________________________________________________
dense (Dense)                (None, 50)                160050    
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 204       
=================================================================
Total params: 217,354
Trainable params: 217,354
Non-trainable params: 0
_________________________________________________________________
```

### 3 of 4
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_26 (Conv2D)           (None, 98, 98, 25)        700       
_________________________________________________________________
dropout_9 (Dropout)          (None, 98, 98, 25)        0         
_________________________________________________________________
max_pooling2d_17 (MaxPooling (None, 49, 49, 25)        0         
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 47, 47, 50)        11300     
_________________________________________________________________
max_pooling2d_18 (MaxPooling (None, 23, 23, 50)        0         
_________________________________________________________________
conv2d_28 (Conv2D)           (None, 21, 21, 50)        22550     
_________________________________________________________________
flatten_12 (Flatten)         (None, 22050)             0         
_________________________________________________________________
dense_27 (Dense)             (None, 50)                1102550   
_________________________________________________________________
dense_28 (Dense)             (None, 4)                 204       
=================================================================
Total params: 1,137,304
Trainable params: 1,137,304
Non-trainable params: 0
_________________________________________________________________
```