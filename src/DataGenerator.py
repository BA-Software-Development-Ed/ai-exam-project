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
        fill_mode='reflect',  # constant, nearest, reflect, wrap
    )

    # private method, returns list of cropped faces from images
    @staticmethod
    def _crop_images(images):
        cropped_images = []

        for image in images:
            cropped_image = DataGenerator.face_detector.crop(image)

            if cropped_image is None:
                continue
            else:
                cropped_images.append(cropped_image)

        return cropped_images

    # returns datasets of auto generated images from image folder
    @staticmethod
    def generate(path, amount, label, test_size):
        file_names = os.listdir(path)
        images = [cv2.imread(f'{path}/{file_name}') for file_name in file_names]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

        cropped_images = DataGenerator._crop_images(images)
        generated_images = []

        for image in cropped_images:
            for i in range(amount):
                generated_image = DataGenerator.datagen.random_transform(image)
                generated_image = generated_image / 255  # normalizing
                generated_images.append([generated_image, label])

        test_size = int(len(generated_images) * test_size)
        train_data, test_data = generated_images[:-test_size], generated_images[-test_size:]

        return (train_data, test_data)

    # returns merged and shuffled datasets
    @staticmethod
    def merge_shuffle(datasets):
        dataset = np.concatenate(datasets)
        dataset = shuffle(dataset)
        data_images, data_labels = dataset[:, 0], dataset[:, 1]

        # voodoo
        data_images = np.array([image for image in data_images], dtype=np.float32)
        data_labels = np.array([label for label in data_labels], dtype=np.float32)

        return (data_images, data_labels)
