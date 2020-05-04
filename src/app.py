from flask import Flask, request, jsonify, render_template
from Utilities import Base64, Files
import time  # for loading simulation
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detection')
def detection():
    return render_template('detection.html')


@app.route('/recognition')
def recognition():
    return render_template('recognition.html')


@app.route('/recognize-profile')
def recognize_profile():
    return render_template('recognize-profile.html')


@app.route('/create-profile')
def create_profile():
    return render_template('create-profile.html')


@app.route('/create-profile', methods=['POST'])
def train():
    response = request.get_json()
    Files.saveBase64(response['images'], 'storage')

    print('respone name;', response['name'])

    data = {'message': f'successfully saved images to memory'}
    time.sleep(5)  # simulates model training
    return jsonify(data), 200


app.run(debug=True)
