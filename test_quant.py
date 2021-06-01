import os
import random

import torch
from torch import quantization
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np
import argparse

import FasterRCNN.quantization_utils as quant_utils
from FasterRCNN.FasterRCNN_MobilenetV2 import FasterRCNN_Generalized,MobileNetV2_Quantized
from FasterRCNN.staircase_dataset import StaircaseDataset
import FasterRCNN.pytorch_vision_utils.utils as utils
from FasterRCNN.pytorch_vision_utils.engine import evaluate

here = os.path.dirname(os.path.abspath(__file__))
# Set environment variables
cuda = torch.cuda.is_available()
random_seed = 0
quant_utils.set_random_seeds(random_seed=random_seed)
cpu_device = torch.device("cpu:0")

# Load quantized model
# out_pth = os.path.join(here,'QAtraining_logs')
# quant_model_filename = 'checkpoint_epoch-0.pth'
out_pth = os.path.join(here,'trained_models')
quant_model_filename = 'FasterRCNN_MobileNetV2_Quantized.pth'
quant_model_filepath = os.path.join(out_pth,quant_model_filename)
quantized_jit_model = quant_utils.load_torchscript_model(model_filepath=quant_model_filepath, device=cpu_device)

# Test data loader
test_root = os.path.join(here,'Dataset','test')
transform = transforms.Compose([transforms.ToTensor()])
kwargs = {}
test_loader = torch.utils.data.DataLoader(StaircaseDataset(test_root,transform,target_transform=None),batch_size=4,
                                            shuffle=True,**kwargs,collate_fn=utils.collate_fn)

# Model COCO evaluation
cuda_false = False
# ev=evaluate(quantized_jit_model, test_loader,cuda_false,cpu_device)

# Load regular model
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
cuda = torch.cuda.is_available()

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
test_loader = torch.utils.data.DataLoader(StaircaseDataset(test_root,transform,target_transform=None),batch_size=4,
                                            shuffle=True,**kwargs,collate_fn=utils.collate_fn)

model_pth = os.path.join(out_pth,'Quantizable_FasterRCNN_MobilenetV2.pth')
device = torch.device('cuda:'+str(1)) if cuda else torch.device('cpu')
model= torch.load(model_pth,map_location=device)

ev=evaluate(model, test_loader,cuda_false,device)
