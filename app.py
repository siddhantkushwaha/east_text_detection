from flask import Flask, request, jsonify
from PIL import Image

import numpy as np
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

    # convert to image to numpy array
    image = np.array(image)
    image = image[:, :, ::-1].copy()

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
    model = load_model(model_path='models/east/model-funsd400.h5')
    graph = tf.get_default_graph()
    app.run(host='0.0.0.0', port=6006)
