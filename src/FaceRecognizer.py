import cv2
import numpy as np
import pandas as pd
from FaceDetector import FaceDetector

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.losses import SparseCategoricalCrossentropy


class FaceRecognizer:

    classes = None
    model = None
    history = None
    faceDetector = FaceDetector('FACE_DEFAULT')

    def __init__(self, classes, model=None):
        self.classes = classes
        if model:
            self.model = model

    # return history, and trains model
    def train_model(self, train_data, test_data, epochs):
        (train_images, train_labels) = train_data
        (test_images, test_labels) = test_data

        loss_func = SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer='adam', loss=loss_func, metrics=['accuracy'])

        history = self.model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

        self.history = pd.DataFrame.from_dict(history.history)
        return self.history

    # returns model evaluation
    def test_model(self, test_images, test_labels):
        self.model.evaluate(test_images, test_labels, verbose=2)

    # saves model and history to directory
    def save_model(self, path):
        self.model.save(f'{path}.h5')
        self.history.to_csv(f'{path}_hist.csv', index=False)

    # loads model and history from directory
    def load_model(self, path):
        self.model = models.load_model(f'{path}.h5')
        self.history = pd.read_csv(f'{path}_hist.csv')

    # returns prediction for largest detection
    def recognize(self, image, gray=False):
        cropped_image = self.faceDetector.crop(image, gray)
        images = np.array([cropped_image], dtype=np.float32)
        normalized_images = np.array([image/255 for image in images], dtype=np.float32)

        predictions = self.model.predict(normalized_images)
        prediction = np.argmax(predictions)
        return self.classes[prediction]

    # returns predictions of cropped images (100 x 100)
    def recognize_many(self, images):
        predictions = self.model.predict(images)
        return predictions

    # prints model summary
    def model_summary(self):
        self.model.summary()

    # returns list of objects with cropped image, face geometrics and predictions
    def face_predictions(self, faces_data, gray=False):
        face_data_images = np.array([face_data['image'] for face_data in faces_data])

        if gray:
            denormalized_images = np.array(face_data_images * 255, dtype=np.uint8)
            face_data_images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in denormalized_images])

        predictions = self.recognize_many(face_data_images)

        for index, value in enumerate(predictions):
            prediction = np.argmax(value)
            faces_data[index]['prediction'] = prediction
            faces_data[index]['predictions'] = value

        return faces_data
