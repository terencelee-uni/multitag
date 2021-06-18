import models
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import ImageDataset
from torch.utils.data import DataLoader

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
#intialize the model
model = models.model(pretrained=False, requires_grad=False).to(device)
# load the model checkpoint
checkpoint = torch.load('../outputs/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

train_csv = pd.read_csv('../input/photos/trainNew.csv')
genres = train_csv.columns.values[2:]
# prepare the test dataset and dataloader
test_data = ImageDataset(
    train_csv, train=False, test=True
)
test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False
)


def preprocess_image(img_path):
    # transformations for the input data
    transforms = Compose([
        transforms.ToPILImage(),
        transforms.Resize((400, 400)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
    ])

    # read input image
    input_img = cv2.imread(img_path)
    # do transformations
    input_data = transforms(image=input_img)["image"]
    # prepare batch
    batch_data = torch.unsqueeze(input_data, 0)

    return batch_data

def postprocess(output_data):
    # get class names
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    # find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    # print the top classes predicted by the model
    while confidences[indices[0][i]] > 0.5:
        class_idx = indices[0][i]
        print(
            "class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item(),
        )
        i += 1

# load pre-trained model -------------------------------------------------------------------------------------------


# preprocessing stage ----------------------------------------------------------------------------------------------
input = preprocess_image("turkish_coffee.jpg").cuda()

# inference stage --------------------------------------------------------------------------------------------------
model.eval()
model.cuda()
output = model(input)

# post-processing stage --------------------------------------------------------------------------------------------
postprocess(output)

# convert to ONNX --------------------------------------------------------------------------------------------------
ONNX_FILE_PATH = "output.onnx"
torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)

onnx_model = onnx.load(ONNX_FILE_PATH)
# check that the model converted fine
onnx.checker.check_model(onnx_model)

print("Model was successfully converted to ONNX format.")
print("It was saved to", ONNX_FILE_PATH)
