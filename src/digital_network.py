import sys
from digital_unit import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import math
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import logging


class DigitalCNN(nn.Module):
    def __init__(self, in_c: int, out_c: int, fc_in: int, fc_out: int = 2):
        super().__init__()
        self.conv_block_1 = ConvBlock(in_c=in_c, out_c=out_c, batch_norm=False, input_complex=False)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=fc_in,
                      out_features=fc_out, dtype=torch.float)
        )

    def forward(self, x):
        x = (self.conv_block_1(x.unsqueeze(dim=-3))) # need to check the sum reduced the dimensions
        x = torch.sum(x, dim=(-1,-2))
        return self.classifier(x)


# train step
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """Performs a training with model trying to learn on data_loader"""
    # training
    train_loss, train_acc = 0, 0
    # Put model into training mode
    model.train()
    # add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        # 1 forward pass
        y_pred = model(X)
        y_prob = nn.functional.softmax(y_pred, dim=1)
        # 2 calculate loss (per batch)
        loss = loss_fn(y_pred, y)    # be very careful here, always use predict results compared to ground truth!!!
        train_loss += loss
        # 3 optimizer zero grad
        optimizer.zero_grad()
        # 4 loss backward
        loss.backward()
        # 5 Optimizer step
        optimizer.step()

        # Calculate accuracy metric
        train_acc += torch.mean((y_pred.argmax(dim=1) == y.argmax(dim=1)).float()).item()
    # Divide total train loss by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    # print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}")
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device):
    """Performs a testing loop step on model going over data_loader"""
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # forward pass
            test_pred = model(X)
            # calculate loss
            test_loss += loss_fn(test_pred, y)
            # calculate acc
            test_pred_labels = test_pred.argmax(dim=1)
            test_acc += (test_pred_labels==y.argmax(dim=1)).sum().item()/len(test_pred_labels)

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}\n")
        return test_loss, test_acc


def train_complex_digital(model: torch.nn.Module,
                          train_dataloader: torch.utils.data.DataLoader,
                          test_dataloader: torch.utils.data.DataLoader,
                          optimizer: torch.optim.Optimizer,
                          loss_fn: torch.nn.Module,
                          epochs: int,
                          device):
    # Create an empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    
    for epoch in range(epochs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_loss, train_acc = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        end.record()
        print(f"Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Train_acc: {train_acc:.4f} | running time: {start.elapsed_time(end) / 1000} s")
        # print(f'Running time for one epoch: {start.elapsed_time(end) / 1000} s')
        # update results dictionary
        results["train_loss"].append(train_loss.cpu().detach().numpy())
        results["train_acc"].append(train_acc)
        # results["test_loss"].append(test_loss.cpu().detach().numpy())
        # results["test_acc"].append(test_acc)
    return results



def test(model: torch.nn.Module,
         loss_fn: torch.nn.Module,
         test_dataloader: torch.utils.data.DataLoader,
         device):
    
    # Create an empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    # pbar = tqdm(range(epochs),file=sys.stdout,position=0,leave=True)
    
    # for epoch in pbar:
    test_loss, test_acc = test_step(model=model,
                                        loss_fn=loss_fn,
                                        data_loader=test_dataloader,
                                        device=device)

        # print out whats happening
        # print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        # pbar.set_description(f"Epoch {epoch+1} | Test loss: {train_loss:.4f} | Test_acc: {train_acc:.4f}")
    results["test_loss"].append(test_loss.cpu().detach().numpy())
    results["test_acc"].append(test_acc)
    # pbar.close()
    # Return the filled results at the end of the epochs
    return results






# model saving functions
def save_digital_model(model, folder_name: str, file_name: str):
    # # create a model directory
    # MODEL_PATH = Path(folder_name)
    # MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = folder_name
    
    # create model save path
    MODEL_NAME = file_name
    MODEL_SAVE_PATH = MODEL_PATH + MODEL_NAME

    # save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

def save_digital_training_curve(model_results, file_path: str):
    # Store data (serialize)
    with open(file_path, 'wb') as handle:
        pickle.dump(model_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saving training curve to: {file_path}")

