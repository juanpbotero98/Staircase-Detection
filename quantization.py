from FasterRCNN.pytorch_vision_utils.coco_eval import evaluate
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
args = parser.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
# Set environment variables
cuda = torch.cuda.is_available()
random_seed = 0
quant_utils.set_random_seeds(random_seed=random_seed)
cuda_device = torch.device("cuda:"+str(args.GPU_ID))
cpu_device = torch.device("cpu:0")

# Load a pretrained model.
model_path = os.path.join(here,args.model_pth)
model = quant_utils.load_model(model_path=model_path,device=cuda_device)
# extract the backbone of the model
backbone = model.backbone
# Move model to CPU
backbone.to(cpu_device)

# # Make a copy of the model for layer fusion
# fused_model = copy.deepcopy(model)

# Redefine RCNN model
quant_backbone = MobileNetV2_Quantized(backbone)
quant_model = FasterRCNN_Generalized(quant_backbone)
quant_model.load_state_dict(model.state_dict(),strict=False)
quant_model.eval()

# Quantification preparation
#quant_utils.fuse_model(quant_model.backbone)
quant_config = torch.quantization.get_default_qconfig("fbgemm")
quant_model.backbone.qconfig = quant_config
torch.quantization.prepare(quant_model, inplace=True) # Perhaps prepare just backbone

# Calibrate model with training data 
train_root = os.path.join(here,'Dataset','train')
transform = transforms.Compose([transforms.ToTensor()])
kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
# Train data loader
train_loader = torch.utils.data.DataLoader(StaircaseDataset(train_root,transform,target_transform=None),batch_size=4,
                                            shuffle=True,**kwargs,collate_fn=utils.collate_fn)
# Calibrate quantization
quant_utils.calibrate_model(model = quant_model, loader = train_loader, device = cpu_device)
quant_model = torch.quantization.convert(quant_model, inplace= True)
quant_utils.print_size_of_model(quant_model)

# Evaluate the quantized model
test_root = os.path.join(here,'Dataset','test')
kwargs = {}
test_loader = torch.utils.data.DataLoader(StaircaseDataset(test_root,transform,target_transform=None),batch_size=4,
                                            shuffle=True,**kwargs,collate_fn=utils.collate_fn)
cuda_false = False
evaluate(quant_model,test_loader,cuda_false,cpu_device)

# Save quantized model
out_pth = os.path.join(here,'trained_models')
quant_model_filename = 'FasterRCNN_MobileNetV2_Quantized.pth'
quant_statedict_filename = 'FasterRCNN_Quantized_StateDict_V2.pth'
quant_utils.save_torchscript_model(model=quant_model, model_dir=out_pth, model_filename=quant_model_filename)
torch.save(quant_model.state_dict(),os.path.join(out_pth,quant_statedict_filename))
# Load quantized model
quant_model_filepath = os.path.join(out_pth,quant_model_filename)
quantized_jit_model = quant_utils.load_torchscript_model(model_filepath=quant_model_filepath, device=cpu_device)