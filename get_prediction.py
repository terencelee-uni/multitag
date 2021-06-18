import json
import pandas as pd
import io
import glob
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch



def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_category(model, tags, image_path):
    with open(image_path, 'rb') as file:
        image_bytes = file.read()

    transformed_image = transform_image(image_bytes=image_bytes)
    outputs = model(transformed_image)
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
    return preds

def get_prediction(model, tags, path_to_directory):
    files = glob.glob(path_to_directory+'/*')
    image_with_tags = {}
    for image_file in files:
        tagList = get_category(model, tags, image_path=image_file)
        joined = ", ".join(tagList)
        image_with_tags[image_file] = joined
    return image_with_tags
