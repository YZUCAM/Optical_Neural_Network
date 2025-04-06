# Stored helper functions used in this package

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from easydict import EasyDict as edict


def tile_kernels(array, num_x, num_y):
    """tile multiple kernels alone x-axis and y-axis
    array should be [C*M*M]
    """
    temp_list = []
    for i in range(num_x):
        temp_list.append(torch.cat([array[i*num_x+j] for j in range(num_y)], -1))
    new_array = torch.cat(temp_list, -2)
    return new_array


def split_kernels(array, num_x, num_y):
    split_row = torch.split(array, array.shape[-2]//num_x, dim=-2)
    new_list = []
    for a in split_row:
        split_col = torch.split(a, a.shape[-1]//num_y, dim=-1)
        new_list += split_col
    return torch.stack(new_list)


def padding(array, whole_dim):
    # pad square array
    array_size = array.shape[-1]
    pad_size1 = (whole_dim-array_size)//2
    pad_size2 = whole_dim-array_size-pad_size1
    array_pad = F.pad(array, (pad_size1, pad_size2, pad_size1, pad_size2))
    return array_pad