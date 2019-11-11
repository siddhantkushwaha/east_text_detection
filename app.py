from flask import Flask, request, jsonify
from PIL import Image

import numpy as np
import cv2
import tensorflow as tf

import logging

from predict import load_model, process_image

app = Flask(__name__)


@app.route('/')
def index():
    return 'Get request to EAST server.'


@app.route('/process', methods=['POST'])
def process():
    image_buf = request.files['image']
    image = Image.open(image_buf).convert('RGB')
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    global graph
    with graph.as_default():
        boxes = process_image(model, image)

    lines = []
    for box in boxes:
        line = box.reshape((8,)).tolist()
        lines.append(line)

    return jsonify(lines)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    model = load_model(model_path='models/east/model-funsd150-icdar200.h5')
    graph = tf.get_default_graph()
    app.run(host='127.0.0.1', port=5001)
