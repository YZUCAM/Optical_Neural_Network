import sys
sys.path.append("/home/imglab/Yi_Zhu/Smart_Project_v10/")
import os
from digital_network import *
from digital_unit import *
from optical_network import *
from optical_unit import *
from torch.utils.data import Dataset, DataLoader, random_split
from new_shapes_dataset import TwoPaddedRotationShapesDataset
import matplotlib.pyplot as plt
import pickle
# from model_retrieve_utils import *
from utils import *

from timeit import default_timer as timer
import time



# hyperparameters for cnn layer
in_c = 1
out_c = 4      # kernels 
fc_in = out_c

# saving configurations
d_server_dir = "/home/imglab/Yi_Zhu/Smart_Project_v10/"
d_sub_dir = "results/"
d_dateinfo = "Mar_01_2024_11_21_23/"

d_model_folder = d_server_dir + d_sub_dir + d_dateinfo +'saved_models/'
d_curve_folder = d_server_dir + d_sub_dir + d_dateinfo +'saved_training_curves/'

d_model_file_name = "complex_valued_digital_cnn_4kernel_ep300_lrp0005.pt"
d_curve_file_name = "complex_valued_digital_cnn_4kernel_ep300_lrp0005.pickle"

d_MODEL_PATH = d_model_folder + d_model_file_name
d_CURVE_PATH = d_curve_folder + d_curve_file_name

digital_model = DigitalCNN(in_c, out_c, fc_in)
digital_model.load_state_dict(torch.load(f=d_MODEL_PATH))

# optical parameter
system_dim = 200
canvas_dim = system_dim // 4
phase_dim = 200
pixel_size = 9.2e-6
focal_length = 20e-3    #  200e-3  # 14.5e-2
wave_lambda = 795e-9

rows = 2
cols = 2

lr=0.001

# saving configurations
o_server_dir = "/home/imglab/Yi_Zhu/Smart_Project_v10/"
o_sub_dir = "results/"
o_dateinfo = "Mar_04_2024_13_20_40/"

o_model_folder = o_server_dir + o_sub_dir + o_dateinfo +'saved_onn_models/'
o_curve_folder = o_server_dir + o_sub_dir + o_dateinfo +'saved_training_curves/'

o_model_file_name = "complex_valued_optical_net_4kernel_ep20_lrp05_focal20mm.pt"
o_curve_file_name = "complex_valued_optical_net_4kernel_ep20_lrp05_focal20mm.pickle"

o_MODEL_PATH = o_model_folder + o_model_file_name
o_CURVE_PATH = o_curve_folder + o_curve_file_name

optical_model = FourierConvComplex(system_dim, phase_dim, pixel_size, focal_length, wave_lambda)
optical_model.load_state_dict(torch.load(f=o_MODEL_PATH))

# loaded saved models  !
onn = optical_model
fc = digital_model.classifier
dr = DetectorLayer(canvas_dim, rows, cols)
sensor = Sensor()   # convert complex field into intensity

OpticalConvModel = OpticalConvModel(onn, fc, dr, sensor)

# load model into GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
OpticalConvModel.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=OpticalConvModel.parameters(), lr=lr)

# turn off grad for optical layer !!!!!
# param_to_turn_off_grad = OpticalConvModel.onn.phase.w_p
# param_to_turn_off_grad.requires_grad = False
param1_to_turn_off_grad = OpticalConvModel.fc[1].weight
param1_to_turn_off_grad.requires_grad = False
param2_to_turn_off_grad = OpticalConvModel.fc[1].bias
param2_to_turn_off_grad.requires_grad = False

#-----------------------------------------------------------------------------------#

# hyperparameters for training and testing set
max_epochs = 50

num_training_set = 50000
num_testing_set = 10000
shape_size = 0
dimension = 200
canvas_color = 0
fill = 1


# define dataset and data loader
train_dataset = TwoPaddedRotationShapesDataset(dimension, shape_size=shape_size, length=num_training_set, canvas_color=canvas_color, fill=fill)
train_loader = DataLoader(train_dataset, batch_size=20, num_workers=6, shuffle=False)

test_dataset = TwoPaddedRotationShapesDataset(dimension, shape_size=shape_size, length=num_testing_set, canvas_color=canvas_color, fill=fill)
test_loader = DataLoader(test_dataset, batch_size=2, num_workers=6, shuffle=False)

print(f"Each training epoch batch size: {len(train_loader)}")
print(f"Each testing epoch batch size: {len(test_loader)}")

#-----------------------------------------------------------------------------------#

# saving configurations
# server_dir = "/home/imglab/Yi_Zhu/Smart_Project_v10/"
# sub_dir = "results/"
now = time.strftime('%b_%d_%Y_%H_%M_%S', time.localtime(time.time()))
folder = o_server_dir + o_sub_dir + now

print('Results will be saved to dir: %s' % folder)
if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(folder+'/saved_onn_finetune_models')
    os.makedirs(folder+'/saved_finetune_training_curves')
    shutil.copy(__file__, folder)
    
model_folder = folder+'/saved_onn_finetune_models/'
curve_folder = folder+'/saved_finetune_training_curves/'

# create dict fro hyper parameters
params = {'system_dim': system_dim,
          'phase_dim': phase_dim,
          'wave_lambda': wave_lambda,
          'focal_length': focal_length,
          'pixel_size': pixel_size,
          'learning rate': lr,
          'epochs': max_epochs,
          'in_c': in_c,
          'out_c': out_c,
          'rows': rows,
          'cols': cols
        }
torch.save(params, folder+'/hyper_params.pt')
print("Hyper parameters have been saved.")

print("------------------Training Start------------------------\n")



if __name__ == "__main__":

    model_file_name = "fine_tune_onn_4kernel_ep50_lrp001_focal20mm.pt"
    curve_file_name = "fine_tune_onn_4kernel_ep50_lrp001_focal20mm.pickle"

    
    start_time = timer()
    
    model_results = train_complex_digital(model=OpticalConvModel,
                                          train_dataloader=train_loader,
                                          test_dataloader=test_loader,
                                          optimizer=optimizer,
                                          loss_fn=loss_fn,
                                          epochs=max_epochs,
                                          device=device)

    end_time = timer()

    print(f"Total training time: {end_time - start_time:.3f} seconds")

    save_model(OpticalConvModel, model_folder, model_file_name)
    save_training_curve(model_results, curve_folder + curve_file_name)

    print(f"Training Completed taking {(end_time - start_time)//60:.2f} minutes")











