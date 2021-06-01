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

#------------------------ Argument parser ------------------------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
parser.add_argument('-g', '--GPU_ID', type=int, default = 0, help='gpu id')
parser.add_argument('--model_pth',type = str, default = os.path.join('trained_models','Quantizable_FasterRCNN_MobilenetV2.pth'),
                    help = 'Path of the trained model')
parser.add_argument('--mode',type= str, default='test',help = 'Run either "demo" or "test" of the final model')
parser.add_argument('--img', type= str, default='XXX.jpg', help = 'Image name (taken from test data) to run the demo')
args = parser.parse_args()
#------------------------ Paths ------------------------
here = os.path.dirname(os.path.abspath(__file__))
trained_models_pth = os.path.join(here,'trained_models')
final_model_fname = 'FasterRCNN_Quantization_Aware.pth' #CHANGE FOR FINAL MODEL
final_model_path = os.path.join(trained_models_pth,final_model_fname)
#------------------------ Load model ------------------------
cpu_device = cpu_device = torch.device("cpu:0")
model = quant_utils.load_torchscript_model(model_filepath=final_model_path)
cuda_false = False

if args.mode == 'demo':
    print('Entering demo mode...')

elif args.mode == 'test':
    print('Entering test mode...')
    # Test data loader
    test_root = os.path.join(here,'Dataset','test')
    if not os.path.exists(test_root):
        print('Test data has not been softlinked or was not softlinked correctly')
        print('Refer to the README file for further instructions')
    transform = transforms.Compose([transforms.ToTensor()])
    kwargs = {}
    test_loader = torch.utils.data.DataLoader(StaircaseDataset(test_root,transform,target_transform=None),batch_size=4,
                                                shuffle=True,**kwargs,collate_fn=utils.collate_fn)
    # Run COCO evaluation
    evaluate(model, test_loader,cuda_false,cpu_device)
    # Model size
    quant_utils.print_size_of_model(model)
    # Inference latency
    input_size = (1,3,320,320)
    quant_latency = quant_utils.measure_inference_latency(model,cpu_device,True,input_size)
    print("Int8 CPU Inference Latency: {:.2f} ms / sample".format(quant_latency* 1000))

else:
    print('Invalid --mode argument, try with "demo" or "test"')
