import sys
sys.path.append("/home/imglab/Yi_Zhu/Smart_Project_v11/")
import os

from utils import *
from optical_unit import *
from optical_network import *
from digital_network import DigitalCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import shutil

from torch.utils.data import Dataset, DataLoader
from new_shapes_dataset import TwoPaddedRotationShapesDataset

from timeit import default_timer as timer
import time


# hyperparameters for training
lr = 0.00005
max_epochs = 2000
batch_epochs = 100

# optical parameter
system_dim = 200
canvas_dim = system_dim // 4
phase_dim = 200
pixel_size = 9.2e-6
focal_length = 20e-3       # 16e-3     # 200e-3     # 14.5e-2
wave_lambda = 795e-9

# hyperparameters for digital cnn layer used to retrieve kernels
in_c = 1
out_c = 4      # kernels 
fc_in = out_c
rows = 2
cols = 2

# prepare psf single kernel from multiple complex valued kernels
server_dir = "/home/imglab/Yi_Zhu/Smart_Project_v11/"
sub_dir = "results/"
dateinfo = "Mar_11_2024_15_20_38/"

model_folder = server_dir + sub_dir + dateinfo +'saved_models/'
model_file_name = "complex_valued_digital_cnn_4kernel_ep600_lrp0005_Imginverse50x50.pt"

MODEL_PATH = model_folder + model_file_name

reloaded_model = DigitalCNN(in_c, out_c, fc_in)
reloaded_model.load_state_dict(torch.load(f=MODEL_PATH))

kernels = reloaded_model.conv_block_1.layers[0].complex_kernel
psf = padding(kernels, canvas_dim*2)
psf_label_nopad = tile_kernels(psf.squeeze(), rows, cols).unsqueeze(0)
psf_label = padding(psf_label_nopad, phase_dim)
print(f"kernels have been loaded from {model_file_name}. Tiled kernels as psf_lab with shape: {psf_label.shape}")

# initialize impulse function
impulse_image = impulse_function(phase_dim)

print(f"Coherent impulse function has been used with shape: {impulse_image.shape}")


# initialize mode
onn = FourierConvComplex(system_dim, phase_dim, pixel_size, focal_length, wave_lambda)

# define loss fn and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
onn.to(device)

# initialize impulse function
impulse_image = impulse_function(phase_dim).to(device)

print(f"Coherent impulse function has been used with shape: {impulse_image.shape}")


loss_fn = MaskLoss(psf_label)
optimizer = torch.optim.Adam(params=onn.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150)

# saving configurations
# server_dir = "/home/imglab/Yi_Zhu/Smart_Project_v10/"
# sub_dir = "results/"
now = time.strftime('%b_%d_%Y_%H_%M_%S', time.localtime(time.time()))
folder = server_dir + sub_dir + now

print('Results will be saved to dir: %s' % folder)
if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(folder+'/saved_onn_models')
    os.makedirs(folder+'/saved_training_curves')
    shutil.copy(__file__, folder)
    
model_folder = folder+'/saved_onn_models/'
curve_folder = folder+'/saved_training_curves/'

# create dict fro hyper parameters
params = {'system_dim': system_dim,
          'phase_dim': phase_dim,
          'wave_lambda': wave_lambda,
          'focal_length': focal_length,
          'pixel_size': pixel_size,
          'learning rate': lr,
          'batch_epochs': batch_epochs,
          'overall_epochs': max_epochs,
          'in_c': in_c,
          'out_c': out_c,
          'rows': rows,
          'cols': cols
        }
torch.save(params, folder+'/hyper_params.pt')
print("Hyper parameters have been saved.")

print("------------------Training Start------------------------\n")

if __name__ == "__main__":

    model_file_name = "complex_valued_optical_net_4kernel_ep2000_lrp00005_focal20mm.pt"
    curve_file_name = "complex_valued_optical_net_4kernel_ep2000_lrp00005_focal20mm.pickle"

    
    start_time = timer()


    impulse_image.cuda()
    # print(f'onn device: {onn.device}')
    # print(f'before trianing impulse device: {impulse_image.device}')
    onn_results = train_complex(impulse_image=impulse_image,
                                psf_label=psf_label,
                                model=onn,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                batch_epochs=batch_epochs,
                                epochs=max_epochs,
                                device=device)

    end_time = timer()

    print(f"Total training time: {end_time - start_time:.3f} seconds")

    save_model(onn, model_folder, model_file_name)
    save_training_curve(onn_results, curve_folder + curve_file_name)

    print(f"Training Completed taking {(end_time - start_time)//60:.2f} minutes")