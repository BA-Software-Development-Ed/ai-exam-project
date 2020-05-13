import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from FaceDetector import FaceDetector
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator:

    face_detector = FaceDetector('FACE_DEFAULT')

    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=10,
        horizontal_flip=True,
        fill_mode='constant'
    )

    @staticmethod
    def _crop_images(images):
        cropped_images = []

        for image in images:
            cropped_image = DataGenerator.face_detector.crop(image)

            if cropped_image is None:
                continue
            else:
                resized_cropped_image = cv2.resize(cropped_image, (100, 100))
                cropped_images.append(resized_cropped_image)

        return cropped_images

    @staticmethod
    def generate(path, amount, label):
        file_names = os.listdir(path)
        images = [cv2.imread(f'{path}/{file_name}').astype(np.float32)/255.0 for file_name in file_names]

        cropped_images = DataGenerator._crop_images(images)
        generated_images = []

        for image in cropped_images:
            for i in range(10):
                generated_image = DataGenerator.datagen.random_transform(image)
                generated_images.append([np.array(generated_image), label])

        return np.array(generated_images)

    @staticmethod
    def merge_shuffle(datasets, test_size):
        data = np.concatenate(datasets)
        data = shuffle(data)

        test_size = int(len(data) * test_size)
        train_data, test_data = data[:-test_size], data[-test_size:]

        train_images, train_labels = train_data[:, 0], train_data[:, 1]
        test_images, test_labels = test_data[:, 0], test_data[:, 1]

        return (train_images, train_labels), (test_images, test_labels)
