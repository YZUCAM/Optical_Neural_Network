import sys
sys.path.append("/home/imglab/Yi_Zhu/Smart_Project_v11/")
import os

from digital_network import *
from digital_unit import *
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
lr = 0.0005
max_epochs = 600

# hyperparameters for training and testing set
num_training_set = 50000
num_testing_set = 10000
shape_size = 0
dimension = 50
canvas_color = 1
fill = 0

# hyperparameters for cnn layer
in_c = 1
out_c = 4      # kernels 
fc_in = out_c

# define dataset and data loader
train_dataset = TwoPaddedRotationShapesDataset(dimension, shape_size=shape_size, length=num_training_set, canvas_color=canvas_color, fill=fill)
train_loader = DataLoader(train_dataset, batch_size=20, num_workers=6, shuffle=False)

test_dataset = TwoPaddedRotationShapesDataset(dimension, shape_size=shape_size, length=num_testing_set, canvas_color=canvas_color, fill=fill)
test_loader = DataLoader(test_dataset, batch_size=2, num_workers=6, shuffle=False)

print(f"Each training epoch batch size: {len(train_loader)}")
print(f"Each testing epoch batch size: {len(test_loader)}")



# instantiate model
model = DigitalCNN(in_c, out_c, fc_in)

# define loss fn and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)



# saving configurations
server_dir = "/home/imglab/Yi_Zhu/Smart_Project_v11/"
sub_dir = "results/"
now = time.strftime('%b_%d_%Y_%H_%M_%S', time.localtime(time.time()))
folder = server_dir + sub_dir + now

print('Results will be saved to dir: %s' % folder)
if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(folder+'/saved_models')
    os.makedirs(folder+'/saved_training_curves')
    shutil.copy(__file__, folder)
    
model_folder = folder+'/saved_models/'
curve_folder = folder+'/saved_training_curves/'

print("------------------Training Start------------------------\n")

if __name__ == "__main__":

    model_file_name = "complex_valued_digital_cnn_4kernel_ep600_lrp0005_Imginverse50x50.pt"
    curve_file_name = "complex_valued_digital_cnn_4kernel_ep600_lrp0005_Imginverse50x50.pickle"

    
    start_time = timer()
    
    model_results = train_complex_digital(model=model,
                                          train_dataloader=train_loader,
                                          test_dataloader=test_loader,
                                          optimizer=optimizer,
                                          loss_fn=loss_fn,
                                          epochs=max_epochs,
                                          device=device)

    end_time = timer()

    print(f"Total training time: {end_time - start_time:.3f} seconds")

    save_digital_model(model, model_folder, model_file_name)
    save_digital_training_curve(model_results, curve_folder + curve_file_name)

    print(f"Training Completed taking {(end_time - start_time)//60:.2f} minutes")

















