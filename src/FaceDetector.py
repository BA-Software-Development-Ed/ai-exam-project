import cv2
import math


class FaceDetector:
    """some info"""

    classifier = None

    classifier_paths = {
        'FACE_ALT': '../cascade-classifiers/haarcascade_frontalface_alt.xml',
        'FACE_ALT2': '../cascade-classifiers/haarcascade_frontalface_alt2.xml',
        'FACE_DEFAULT': '../cascade-classifiers/haarcascade_frontalface_default.xml',
    }

    def __init__(self, classifier_type):
        classifier_path = self.classifier_paths[classifier_type]
        self.classifier = cv2.CascadeClassifier(classifier_path)

    # private method, returns face geometrics with biggest diagonal
    def _detect(self, image, gray=False):
        if not gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.classifier.detectMultiScale(image)
        if len(detected_faces) == 0:
            return

        tmp_diagonal = 0
        tmp_face = None
        for (x, y, w, h) in detected_faces:
            diagonal = math.sqrt(w ** 2 + h ** 2)

            if diagonal > tmp_diagonal:
                tmp_diagonal = diagonal
                tmp_face = (x, y, w, h)

        return tmp_face

    # returns cropped and resized face as image (100 x 100)
    def crop(self, image, gray=False):
        face = self._detect(image, gray)
        if not face:
            return

        (x, y, w, h) = face
        cropped_image = image[y:y+h, x:x+w]
        resized_cropped_image = cv2.resize(cropped_image, (100, 100))
        return resized_cropped_image

    # returns list of cropped and resized faces from image
    def crop_all(self, image):
        image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.classifier.detectMultiScale(image_grayscaled)

        if len(detected_faces) == 0:
            return

        faces = []

        for (x, y, w, h) in detected_faces:
            cropped_image = image[y:y+h, x:x+w]
            resized_cropped_image = cv2.resize(cropped_image, (100, 100))
            faces.append(resized_cropped_image)

        return faces

    # returns image marked with biggest detection
    def mark(self, image):
        face = self._detect(image)
        if not face:
            return

        (x, y, w, h) = face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return image

    # returns list of objects with cropped image and face geometrics
    def face_details(self, image):
        image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.classifier.detectMultiScale(image_grayscaled)

        if len(detected_faces) == 0:
            return

        faces_data = []

        for (x, y, w, h) in detected_faces:
            cropped_image = image[y:y+h, x:x+w]
            resized_cropped_image = cv2.resize(cropped_image, (100, 100))
            normalized_image = resized_cropped_image / 255  # normalizing
            face_data = {'image': normalized_image, 'face': (x, y, w, h)}
            faces_data.append(face_data)

        return faces_data

    # returns image with all detections marked
    def mark_all(self, image):
        image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.classifier.detectMultiScale(image_grayscaled)

        for (x, y, w, h) in detected_faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return image
