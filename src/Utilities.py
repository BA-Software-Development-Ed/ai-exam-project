import numpy as np
import cv2
import base64
import seaborn as sns
from matplotlib import pyplot as plt

import tensorflow as tf


class Base64:
    @staticmethod
    def encode():  # encodeArray(), encodeImage()
        pass

    @staticmethod
    def decode(base64String):
        image = base64.b64decode(base64String)
        return image

    @staticmethod
    def decodeAsArray(base64String):
        binary = Base64.decode(base64String)
        image_array = np.asarray(bytearray(binary), dtype="uint8")
        return image_array

    @staticmethod
    def decodeAsImage(base64String):
        image_array = Base64.decodeAsArray(base64String)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image


class Files:
    @staticmethod
    def saveBase64(images, path):
        for index, base64_image in enumerate(images):
            image = Base64.decode(base64_image)
            file_path = f'{path}/image_{index}.jpg'

            with open(file_path, 'wb') as file:
                file.write(image)

    @staticmethod
    def loadImages(path):
        pass


class Displayer:
    @staticmethod
    def image(image, bgr=False):
        if bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.axis("off")
        plt.imshow(image)
        plt.show()

    @staticmethod
    def images(images, amount):
        plt.figure(figsize=(10, 10))

        for index, image in enumerate(images[:amount]):
            plt.axis("off")
            plt.subplot(amount // 5, 5, 1+index)
            plt.imshow(image)

        plt.axis("off")
        plt.show()

    # missing x & y lables
    @staticmethod
    def conf_matrix(predictions, labels):
        predictions = [np.argmax(prediction) for prediction in predictions]
        conf_matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()

        matrix = np.around([row/sum(row) for row in conf_matrix], decimals=2)
        plot = sns.heatmap(matrix, cmap=sns.color_palette("Blues"), annot=True)
        figure = plot.get_figure()

    @staticmethod
    def acc_history(history):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')


class OneHot:
    @staticmethod
    def encode():
        pass

    @staticmethod
    def decode():
        pass
