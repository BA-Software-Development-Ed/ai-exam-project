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

    def train_model(self, train_data, test_data, epochs):
        (train_images, train_labels) = train_data
        (test_images, test_labels) = test_data

        loss_func = SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer='adam', loss=loss_func, metrics=['accuracy'])

        history = self.model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

        self.history = pd.DataFrame.from_dict(history.history)
        return self.history

    def test_model(self, test_images, test_labels):
        self.model.evaluate(test_images, test_labels, verbose=2)

    def save_model(self, path):
        self.model.save(f'{path}.h5')
        self.history.to_csv(f'{path}_hist.csv', index=False)

    def load_model(self, path):
        self.model = models.load_model(f'{path}.h5')
        self.history = pd.read_csv(f'{path}_hist.csv')

    def recognize(self, image, gray=False):
        cropped_image = self.faceDetector.crop(image, gray)
        images = np.array([cropped_image], dtype=np.float32)
        normalized_images = np.array([image/255 for image in images], dtype=np.float32)

        predictions = self.model.predict(normalized_images)
        prediction = np.argmax(predictions)
        return self.classes[prediction]

    def recognize_many(self, images):
        predictions = self.model.predict(images)
        return predictions

    def model_summary(self):
        self.model.summary()

    def face_predictions(self, faces_data):
        face_data_images = np.array([face_data['image'] for face_data in faces_data])
        predictions = self.recognize_many(face_data_images)

        for index, value in enumerate(predictions):
            prediction = np.argmax(value)
            faces_data[index]['prediction'] = prediction

        return faces_data
