import io
import json
import models
import numpy as np
from torch.utils.data import DataLoader
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, jsonify, request

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

train_csv = pd.read_csv('../../trainNew.csv')
tags = train_csv.columns.values[2:]

app = Flask(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
#intialize the model
model = models.model(pretrained=False, requires_grad=False).to(device)
# load the model checkpoint
checkpoint = torch.load('../../model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()
                                        ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model(tensor.cuda())
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    sorted_indices = np.argsort(outputs[0])
    preds = []
    best = sorted_indices[-3:]
    string_predicted = ''
    string_actual = ''
    for i in range(len(best)):
        string_predicted += f"{tags[best[i]]}    "
        preds.append(tags[best[i]])
    print("test")
    return preds

def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is None or file.filename =="":
            return jsonify({'error': 'no file'})
        if not allowed(file.filename):
            return jsonify({'error': 'format not supported'})
        try:
            img_bytes = file.read()
            class_name = get_prediction(image_bytes=img_bytes)
            return jsonify({'class_name': class_name})
        except:
            return jsonify({'error': 'error during prediction'})


if __name__ == '__main__':
    app.run()
