from optical_unit import AngSpecProp, Lens, PhaseMask, LightPropagation
from utils import padding, tile_kernels, split_kernels
import shutil
import time
import numpy as np
import os
import torch
from torchvision import transforms
import math
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from utils import *


class FourierConv(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda, weights=None):
        super(FourierConv, self).__init__()
        self.prop = AngSpecProp(
            whole_dim, pixel_size, focal_length, wave_lambda)
        self.lens = Lens(whole_dim, pixel_size, focal_length, wave_lambda)

        phase_init = weights["phase"] if weights else None
        self.phase = PhaseMask(whole_dim, phase_dim, phase=phase_init)

        scalar_init = weights["scalar"] if weights else None
        self.scalar = torch.ones(1, dtype=torch.float32)*0.1 if scalar_init is None else torch.tensor(
            weights["scalar"], dtype=torch.float32)
        self.w_scalar = nn.Parameter(self.scalar)

    def forward(self, input_field):
        x = self.prop(input_field)
        x = self.lens(x)
        x = self.prop(x)
        x = self.phase(x)
        x = self.prop(x)
        x = self.lens(x)
        x = self.prop(x)
        output = torch.abs(self.w_scalar*x)**2
        return output



class FourierConvComplex(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda, weights=None):
        super(FourierConvComplex, self).__init__()
        self.prop = AngSpecProp(whole_dim, pixel_size, focal_length, wave_lambda)
        # self.prop = LightPropagation(whole_dim, whole_dim, pixel_size, wave_lambda, focal_length)
        self.lens = Lens(whole_dim, pixel_size, focal_length, wave_lambda)

        phase_init = weights["phase"] if weights else None
        self.phase = PhaseMask(whole_dim, phase_dim, phase=phase_init)

        scalar_init = weights["scalar"] if weights else None
        self.scalar = torch.ones(1, dtype=torch.float32)*1 if scalar_init is None else torch.tensor(
            weights["scalar"], dtype=torch.float32)
        self.w_scalar = nn.Parameter(self.scalar)

    def forward(self, input_field):
        x = self.prop(input_field)
        x = self.lens(x)
        x = self.prop(x)
        x = self.phase(x)
        x = self.prop(x)
        x = self.lens(x)
        x = self.prop(x)
        output = self.w_scalar*x
        return output


class MaskLoss(nn.Module):
    def __init__(self, target):
        super(MaskLoss, self).__init__()
        self.target =target.cuda()

    def forward(self, input_field):
        diff1 = (input_field.real - self.target.real)
        diff2 = (input_field.imag - self.target.imag)
        return torch.mean(diff1.pow(2)+diff2.pow(2))
        # return torch.mean(torch.abs(diff)**2)
        

class DetectorLayer(nn.Module):
    def __init__(self, dimension_image, rows, cols):
        super(DetectorLayer, self).__init__()
        self.dim_image = dimension_image
        self.rows = rows
        self.cols = cols
        self.crop = transforms.CenterCrop(self.dim_image)

    def forward(self,input_field):
        fields = split_kernels(input_field, self.rows, self.cols)
        fields_crop = self.crop(fields)
        return fields_crop.permute(1,0,2,3)
        

'''used for fine tune the fc layer!'''
class OpticalConvModel(nn.Module):
    def __init__(self, onn, fc, dr, sensor):
        super(OpticalConvModel, self).__init__()

        self.onn = onn
        self.fc = fc
        self.dr = dr
        self.sensor = sensor

    def forward(self, x):
        x = self.onn(x)
        x = self.dr(x)
        x = self.sensor(x)
        x = torch.sum(x, dim=(-1,-2))
        x = self.fc(x)
        return x






def impulse_function(dimension):
    image = np.zeros((1, dimension, dimension))
    image[0, dimension//2, dimension//2] = 1
    image = torch.tensor(image, dtype=torch.complex64)
    return image


# def cropped_loss(output, target):
#     diff = (output-target)
#     return torch.mean(torch.abs(diff)**2)



# train setup
def train_step(impulse_image: torch.tensor,
               psf_label: torch.tensor,
               model: torch.nn.Module,
               loss_fn,
               optimizer: torch.optim.Optimizer,
               batch_epochs: int,
               device: torch.device):
    """Performs a training with model trying to learn on data_loader"""

    # training
    train_loss, train_acc = 0, 0
    # Put required tensor into GPU
    model.to(device)
    impulse_image.to(device)
    # psf_label = padding(psf_label, dim)
    psf_label.to(device)
    # Put model into training mode
    model.train()

    for epoch in range(batch_epochs):
        # Compute prediction and loss
        pred = model(impulse_image)
        loss = loss_fn(pred)
        train_loss += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss = loss.item()
        # scheduler.step(loss)

        # print(f"epoch: {epoch} loss: {loss:>8f} Current learning rate: {optimizer.param_groups[0]['lr']}\r\n")
        # if epoch % 1000 == 0:
        #     print(f"inner epoch: {epoch} loss: {loss:>8f} Current learning rate: {optimizer.param_groups[0]['lr']}")

    # Divide total train loss by length of train dataloader
    train_loss /= batch_epochs
    return train_loss


def train_complex(impulse_image: torch.tensor,
                  psf_label: torch.tensor,
                  model: torch.nn.Module,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  batch_epochs: int,
                  epochs: int,
                  device: torch.device):

    # Create an empty results dictionary
    results = {"train_loss": []}

    for ep in range(epochs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_loss = train_step(impulse_image=impulse_image,
                                           psf_label=psf_label,
                                           model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           batch_epochs = batch_epochs,
                                           device=device)
        end.record()
        print(f"Epoch {ep+1}/{epochs} | Train loss: {train_loss:.5f} | running {batch_epochs} iterates takes {start.elapsed_time(end) / 1000} s")
        # update results dictionary
        results["train_loss"].append(train_loss.cpu().detach().numpy())
    return results


# model saving functions
def save_model(model, folder_name: str, file_name: str):
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

def save_training_curve(model_results, file_path: str):
    # Store data (serialize)
    with open(file_path, 'wb') as handle:
        pickle.dump(model_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saving training curve to: {file_path}")
 













