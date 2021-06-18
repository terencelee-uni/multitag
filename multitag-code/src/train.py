CUDA_LAUNCH_BLOCKING="1"
import models
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from engine import train, validate
from dataset import ImageDataset
from torch.utils.data import DataLoader
matplotlib.style.use('ggplot')
# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#intialize the model
model = models.model(pretrained=True, requires_grad=False).to(device)
# learning parameters
lr = 0.0001
epochs = 1
batch_size = 20
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# read the training csv file
train_csv = pd.read_csv('../input/photos/trainNew.csv')
# train dataset
train_data = ImageDataset(
    train_csv, train=True, test=False
)
# validation dataset
valid_data = ImageDataset(
    train_csv, train=False, test=False
)
# train data loader
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
# validation data loader
valid_loader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=False
)

# start the training and validation
train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_loader, optimizer, criterion, train_data, device
    )
    valid_epoch_loss = validate(
        model, valid_loader, criterion, valid_data, device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {valid_epoch_loss:.4f}')

    # save the trained model to disk
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, '../outputs/model.pth')

# state_dict = torch.load('../outputs/model.pth')
# model.load_state_dict(state_dict)
# dummy_input = torch.randn(1, 3, 400, 400)
#
# torch.onnx.export(model, dummy_input, "onnx_model_name.onnx")

# plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../outputs/loss.png')
plt.show()

# import torch
# import cv2
# import numpy as np
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset

# def preprocess_image(img_path):
#     # transformations for the input data
#     transforms = Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((400, 400)),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomRotation(degrees=45),
#                 transforms.ToTensor(),
#     ])
#
#     # read input image
#     input_img = cv2.imread(img_path)
#     # do transformations
#     input_data = transforms(image=input_img)["image"]
#     # prepare batch
#     batch_data = torch.unsqueeze(input_data, 0)
#
#     return batch_data
#
# def postprocess(output_data):
#     # get class names
#     with open("tags.txt") as f:
#         classes = [line.strip() for line in f.readlines()]
#     # calculate human-readable value by softmax
#     confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
#     # find top predicted classes
#     _, indices = torch.sort(output_data, descending=True)
#     i = 0
#     # print the top classes predicted by the model
#     while confidences[indices[0][i]] > 0.5:
#         class_idx = indices[0][i]
#         print(
#             "class:",
#             classes[class_idx],
#             ", confidence:",
#             confidences[class_idx].item(),
#             "%, index:",
#             class_idx.item(),
#         )
#         i += 1
#
#     # preprocessing stage ---------------------------------------
# input = preprocess_image("scenery.jpg").cuda()
#
#     # inference stage -------------------------------------------
# model.eval()
# model.cuda()
# output = model(input)
#
#     # post-processing stage -------------------------------------
# postprocess(output)

#     # convert to ONNX -------------------------------------------
# ONNX_FILE_PATH = "resnet50.onnx"
# torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)
#
# onnx_model = onnx.load(ONNX_FILE_PATH)
#     # check that the model converted fine
# onnx.checker.check_model(onnx_model)
#
# print("Model was successfully converted to ONNX format.")
# print("It was saved to", ONNX_FILE_PATH)
