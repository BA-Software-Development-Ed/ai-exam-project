from flask import Flask, request, render_template, jsonify
import cv2
import base64
import numpy as np
import os
import io
from PIL import Image

from Utilities import Base64, Files

from FaceDetector import FaceDetector
from FaceRecognizer import FaceRecognizer

app = Flask(__name__)
classes = ['dad', 'mom', 'son', 'daughter']

faceDetector = FaceDetector('FACE_ALT')
faceRecognizer = FaceRecognizer(classes, faceDetector)
faceRecognizer.load_model('src/models/cnn_model_3of4')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detection',  methods=["GET", "POST"])
def detection():
    if request.method == "GET":
        return render_template('detection.html')

    # get base64 image from request
    raw_image = request.get_json()['image']

    # decodes base64 image to cv image
    image = Base64.decodeAsImage(raw_image)

    # detects faces on image
    marked_image = faceDetector.mark_all(image)

    img = Image.fromarray(marked_image.astype("uint8"))

    # voodoo
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)

    # ...
    img_base64 = base64.b64encode(rawBytes.read())
    response = str(img_base64, 'utf-8')

    return jsonify({'image': response}), 200


@app.route('/recognition',  methods=["GET", "POST"])
def recognition():

    if request.method == "GET":
        return render_template('recognition.html')

    # get base64 image from request
    raw_image = request.get_json()['image']

    # decodes base64 image to cv image
    image = Base64.decodeAsImage(raw_image)

    # detects faces on image
    faces_data = faceDetector.face_details(image)
    prediction_data = faceRecognizer.face_predictions(faces_data)

    # draws face boxes and names on image
    for face_data in prediction_data:
        (x, y, w, h) = face_data['face']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 125), 2)
        cv2.putText(image, classes[face_data['prediction']], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 125), 2)

    img = Image.fromarray(image.astype("uint8"))

    # voodoo
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)

    # ...
    img_base64 = base64.b64encode(rawBytes.read())
    response = str(img_base64, 'utf-8')

    return jsonify({'image': response}), 200


@app.route('/create-profile',  methods=["GET", "POST"])
def create_profile():
    if request.method == "GET":
        return render_template('create-profile.html')

    # get base64 content
    return jsonify({'status': 'not implemented'}), 501


@app.route('/recognize-profile',  methods=["POST"])
def recognize_profile():

    # get base64 content
    return jsonify({'status': 'not implemented'}), 501


app.run(debug=True)
