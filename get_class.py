from flask import Flask, render_template, request, redirect, url_for
from get_images import get_images, get_path, get_directory
from get_prediction import get_prediction
from generate_html import generate_html
import torch
import models
import pandas as pd
import json

app = Flask(__name__)

# class_mapping = json.load(open('vacationTags.json'))

device = torch.device("cpu")
# device = torch.device("cpu")
#intialize the model
model = models.model(pretrained=False, requires_grad=False).to(device)
# load the model checkpoint
checkpoint = torch.load('./model.pth', map_location=torch.device('cpu'))
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def get_image_class(path):
    get_images(path)
    path = get_path(path)
    train_csv = pd.read_csv('./trainNew.csv')
    tags = train_csv.columns.values[2:]
    images_with_tags = get_prediction(model, tags, path)
    print(type(images_with_tags))
    generate_html(images_with_tags)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        get_image_class(user)
        return redirect(url_for('success', name=get_directory(user)))


@app.route('/success/<name>')
def success(name):
    return render_template('image_class.html')


if __name__ == '__main__' :
    app.run(debug=True)
