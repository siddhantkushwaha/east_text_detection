import logging

from flask import Flask, request
import numpy as np
from PIL import Image

from predict import load_model, process_image

app = Flask(__name__)


@app.route('/')
def index():
    return 'Get request to EAST server.'


@app.route('/process', methods=['POST'])
def process():
    image_buf = request.files['image']
    image = Image.open(image_buf)

    # convert to image to numpy array
    image = np.array(image)
    print(image.shape)

    boxes = process_image(model, image)
    print(boxes)

    return 'Ok.'


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    model = load_model(model_path='models/east/model-funsd400.h5')
    app.run(host='0.0.0.0', port=6006)
