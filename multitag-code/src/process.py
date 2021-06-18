import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from PIL import Image

# Filter harmless warnings
import warnings
warnings.filterwarnings("ignore")

path = '..\\input\\photos\\Images\\'
img_names = []
tp = []

for folder, subfolders, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder+'\\'+img)
        tp.append(img)

print('Images: ',len(img_names))


img_sizes = []
rejected = []

for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)

print(f'Images:  {len(img_sizes)}')
print(f'Rejects: {len(rejected)}')

# Convert the list to a DataFrame
df = pd.DataFrame(img_sizes)

# Run summary statistics on image widths
print(df[0].describe())
print(df[1].describe())

transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
for idx, val in enumerate(img_names):
    value = tp[idx][tp[idx].find("(")+1:tp[idx].find(")")]
    temp = img_names[idx].replace("Images", "ImageProcessed")
    string = str(idx + 1)
    temp2 = temp.replace(value, string)
    y = transform(Image.open(val))
    z = transforms.ToPILImage()(y)
    z.save(temp2, 'JPEG')
