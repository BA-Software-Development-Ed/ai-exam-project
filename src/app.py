from flask import Flask, request, redirect, render_template, jsonify
import cv2
import base64
import numpy as np
import os
import io
from PIL import Image

from Utilities import Base64, Files

from FaceDetector import FaceDetector

app = Flask(__name__)

faceDetector = FaceDetector('FACE_ALT2')


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

    raw_image = request.files["image"].read()
    np_image = np.fromstring(raw_image, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)

    return 200


app.run(debug=True)
