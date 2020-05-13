import cv2


class FaceDetector:
    """Some info"""

    classifier = None

    classifier_paths = {
        'CAT_EXTENDED': '../cascade-classifiers/haarcascade_frontalcatface_extended.xml',
        'CAT': '../cascade-classifiers/haarcascade_frontalcatface.xml',
        'FACE_ALT': '../cascade-classifiers/haarcascade_frontalface_alt.xml',         # works!
        'FACE_ALT2': '../cascade-classifiers/haarcascade_frontalface_alt2.xml',       # works!
        'FACE_DEFAULT': '../cascade-classifiers/haarcascade_frontalface_default.xml',  # works best!
    }

    def __init__(self, classifier_type):
        classifier_path = self.classifier_paths[classifier_type]
        self.classifier = cv2.CascadeClassifier(classifier_path)

    def crop(self, image):
        image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.classifier.detectMultiScale(image_grayscaled)
        (x, y, w, h) = detected_faces[0]
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image

    def detect(self, image):
        image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.classifier.detectMultiScale(image_grayscaled)
        (x, y, w, h) = detected_faces[0]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return image
