import matplotlib.pyplot as plt
import numpy as np
# import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageChops
from torch.utils.data import DataLoader, Dataset
import math
import random

################################################################################
# consider the simulation window is 10 mm                                      #
# the object dimension is 5 mm, there will be 5mm guarding regime in each side #
################################################################################
# input dimension sould be 1000, dx = pixel_size = 9.2e-6, shape size range from 1 to 2 mm
# class TwoPaddedShapesDataset(Dataset):
#     def __init__(self, input_dimension, shape_size=20, length=1000):
#         self.input_dimension = input_dimension
#         self.length = length
#         # self.shape_size = shape_size
#         # self.shape_size = random.randint(10, 150)
#         self.canvas_size = 200 # input_dimension//5         # (dimension - canvas_size)%2 = 0

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         # self.shape_size = random.randint(20, 150)
#         self.shape_size = random.randint(20, 80)
#         # self.shape_size = 50
#         image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)  # Create a black image
#         draw = ImageDraw.Draw(image)

#         # Decide whether to draw a square or a circle (0 for square, 1 for circle)
#         label = torch.randint(0, 2, size=(1,)).item()
#         shape_center = (
#             np.random.randint(0, self.canvas_size - self.shape_size/2),
#             np.random.randint(0, self.canvas_size - self.shape_size/2),
#         )

#         # Convert label to one-hot encoding
#         # label_one_hot = torch.zeros(10)
        
#         if label == 0: 
#             # Draw circle 
#             draw.ellipse([shape_center, (shape_center[0] + self.shape_size, shape_center[1] + self.shape_size)], fill=10)
#         else:  
#             # Draw square
#             draw.rectangle([shape_center, (shape_center[0] + self.shape_size, shape_center[1] + self.shape_size)], fill=10)            

#         image = transforms.Pad((self.input_dimension - self.canvas_size)//2)(image)
#         image = transforms.ToTensor()(image).squeeze()
#         image = image / image.max()
#         # image = image / image.abs().pow(2).sum().sqrt()
#         # Normalize by subtracting mean and dividing by standard deviation
#         # mean = np.mean(image, axis=(0, 1))
#         # std = np.std(image, axis=(0, 1))
#         # image = (image - mean) / std|
#         # image = torch.nn.functional.normalize(image, (0.1307,), (0.3081,))
#         # Convert label to one-hot encoding
#         label_one_hot = torch.zeros(2)
#         label_one_hot[label] = 1
#         return image, label_one_hot.float()


class TwoPaddedRotationShapesDataset(Dataset):
    def __init__(self, input_dimension, shape_size=20, length=1000, canvas_color=0, fill=1):
        self.input_dimension = input_dimension
        self.shape_size = shape_size
        self.length = length
        # self.shape_size = random.randint(10, 150)
        self.canvas_size = self.input_dimension          # input_dimension//4   # (dimension - canvas_size)%2 = 0
        self.canvas_color = canvas_color
        self.fill = fill
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.shape_size:
            self.shape_size = random.randint(int(self.canvas_size*0.2), int(self.canvas_size*0.8))
        image = Image.new("L", (self.canvas_size, self.canvas_size), color=self.canvas_color)  # Create a black image
        draw = ImageDraw.Draw(image)

        # Decide whether to draw a square or a circle (0 for square, 1 for circle)
        label = torch.randint(0, 2, size=(1,)).item()
        shape_center = (
            np.random.randint(0, self.canvas_size - self.shape_size/2),
            np.random.randint(0, self.canvas_size - self.shape_size/2),
        )

        random_angle = np.random.randint(0, 90)

        # Convert label to one-hot encoding
        # label_one_hot = torch.zeros(10)
        
        if label == 0: 
            # Draw circle 
            draw.ellipse([shape_center, (shape_center[0] + self.shape_size, shape_center[1] + self.shape_size)], fill=self.fill)
        else:  
            # Draw square
            square_vertices = ((shape_center[0] + self.shape_size / 2, shape_center[1] + self.shape_size / 2),
                               (shape_center[0] + self.shape_size / 2, shape_center[1] - self.shape_size / 2),
                               (shape_center[0] - self.shape_size / 2, shape_center[1] - self.shape_size / 2),
                               (shape_center[0] - self.shape_size / 2, shape_center[1] + self.shape_size / 2))
            
            square_vertices = [self.rotated_about(x,y, shape_center[0], shape_center[1], math.radians(random_angle)) for x,y in square_vertices]
            
            # draw.rectangle([shape_center, (shape_center[0] + self.shape_size, shape_center[1] + self.shape_size)], fill=10)  
            draw.polygon(square_vertices, fill=self.fill)

        image = transforms.Pad((self.input_dimension - self.canvas_size)//2)(image)
        image = transforms.ToTensor()(image).squeeze()
        image = image / image.max()
        # image = image / image.abs().pow(2).sum().sqrt()
        # Normalize by subtracting mean and dividing by standard deviation
        # mean = np.mean(image, axis=(0, 1))
        # std = np.std(image, axis=(0, 1))
        # image = (image - mean) / std|
        # image = torch.nn.functional.normalize(image, (0.1307,), (0.3081,))
        # Convert label to one-hot encoding
        label_one_hot = torch.zeros(2)
        label_one_hot[label] = 1
        return image, label_one_hot.float()


    #finds the straight-line distance between two points
    def distance(self, ax, ay, bx, by):
        return math.sqrt((by - ay)**2 + (bx - ax)**2)

    #rotates point `A` about point `B` by `angle` radians clockwise.
    def rotated_about(self, ax, ay, bx, by, angle):
        radius = self.distance(ax,ay,bx,by)
        angle += math.atan2(ay-by, ax-bx)
        return (round(bx + radius * math.cos(angle)),
                round(by + radius * math.sin(angle)))


















        

# # 16 hot code
# class TwoPaddedShapesDataset16(Dataset):
#     def __init__(self, input_dimension, shape_size=20, length=1000):
#         self.input_dimension = input_dimension
#         self.length = length
#         # self.shape_size = shape_size
#         # self.shape_size = random.randint(10, 150)
#         self.canvas_size = 100

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         # self.shape_size = random.randint(100, 220)
#         self.shape_size = 25
#         image = Image.new("L", (self.canvas_size, self.canvas_size), color=1)  # Create a black image
#         draw = ImageDraw.Draw(image)

#         # Decide whether to draw a square or a circle (0 for square, 1 for circle)
#         label = torch.randint(0, 2, size=(1,)).item()
#         shape_center = (
#             np.random.randint(0, self.canvas_size - self.shape_size),
#             np.random.randint(0, self.canvas_size - self.shape_size),
#         )

#         # Convert label to one-hot encoding
#         label_one_hot = torch.zeros(16)
        
#         if label == 0: 
#             # Draw circle 
#             draw.ellipse([shape_center, (shape_center[0] + self.shape_size, shape_center[1] + self.shape_size)], fill=10)
#             label_one_hot[label] = 1
#         else:  
#             # Draw square
#             draw.rectangle([shape_center, (shape_center[0] + self.shape_size, shape_center[1] + self.shape_size)], fill=10)
#             label_one_hot[label+10] = 1

#         image = transforms.Pad((self.input_dimension - self.canvas_size)//2)(image)
#         image = transforms.ToTensor()(image).squeeze()
#         image = image / image.max()
#         # image = image / image.abs().pow(2).sum().sqrt()
#         # Normalize by subtracting mean and dividing by standard deviation
#         # mean = np.mean(image, axis=(0, 1))
#         # std = np.std(image, axis=(0, 1))
#         # image = (image - mean) / std|
#         # image = torch.nn.functional.normalize(image, (0.1307,), (0.3081,))
#         # Convert label to one-hot encoding
#         # label_one_hot = torch.zeros(2)
#         # label_one_hot[label] = 1
#         return image, label_one_hot.float()
