import io
import json
import app.models
import numpy as np
from torch.utils.data import DataLoader
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os

from flask import Flask, render_template, request, redirect, jsonify

from functions import transform_image, get_prediction, allowed

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        return render_template('result.html',
                               class_name=class_name)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
