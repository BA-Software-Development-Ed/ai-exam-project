import numpy as np
import cv2
import base64


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


class Chart:
    @staticmethod
    def image(image):
        pass

    @staticmethod
    def heatmap(data):
        pass


class OneHot:
    @staticmethod
    def encode():
        pass

    @staticmethod
    def decode():
        pass
