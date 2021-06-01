import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
from .mobilenetv2 import ConvBNReLU,InvertedResidual

def load_model(model_path, device):
    model = torch.load(model_path,map_location=device)
    return model

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for images, labels in tqdm(loader,desc= 'Calibrating model quantization'):
        images = list(img.to(device) for img in images)
        _ = model(images)

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model

def convert_jit2statedict(mode_filepath,device,model_convert_path):
    model = load_torchscript_model(mode_filepath,device)


def get_flops(model,device,input_size):
    model.to(device)
    model.eval()
    inputs = torch.rand(size=input_size).to(device)
    flops = FlopCountAnalysis(model,inputs)
    total_flops = flops.total()
    return total_flops

def measure_inference_latency(model,
                              device,quant=False,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    
    if quant:
        x = torch.rand(size=input_size[1:]).to(device) 
        x=[x]
    else: 
        x = torch.rand(size=input_size).to(device)    
    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    #torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            #torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave

def fuse_model(self):
    for m in self.modules():
        if type(m) == ConvBNReLU:
            torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
        if type(m) == InvertedResidual:
            for idx in range(len(m.conv)):
                if type(m.conv[idx]) == nn.Conv2d:
                    torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')