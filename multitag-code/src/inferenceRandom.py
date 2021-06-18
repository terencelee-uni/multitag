import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import ImageDataset
from torch.utils.data import DataLoader
import json
with torch.no_grad():
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

    train_csv = pd.read_csv('../input/photos/shuffled.csv')
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

    predList = []
    for counter, data in enumerate(test_loader):
        image, target = data['image'].to(device), data['label']
        # get all the index positions where value == 1
        target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
        # get the predictions by passing the image through the model

        outputs = model(image)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.detach().cpu()
        sorted_indices = np.argsort(outputs[0])
        preds = []
        best = sorted_indices[-3:]
        string_predicted = ''
        string_actual = ''
        for i in range(len(best)):
            string_predicted += f"{genres[best[i]]}    "
            preds.append(genres[best[i]])
        for i in range(len(target_indices)):
            string_actual += f"{genres[target_indices[i]]}    "
        predList.append(preds)
        image = image.squeeze(0)
        image = image.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
        plt.savefig(f"../outputs/inference_{counter}.jpg")
        plt.show()
        predList.append(preds)

