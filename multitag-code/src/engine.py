CUDA_LAUNCH_BLOCKING="1"
import torch
from tqdm import tqdm

# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        # print(outputs)
        # print(target)
        loss = criterion(outputs + 1e-10, target + 1e-10)
        if torch.isnan(loss):
            counter -= 1
            continue
        train_running_loss += loss.item()
        # backpropagation
        loss.backward()
        # update optimizer parameters
        clipping_value = 1 # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss

    # validation function
def validate(model, dataloader, criterion, val_data, device):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()

        val_loss = val_running_loss / counter
        return val_loss
