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
from PIL import Image
import xmltodict
import cv2
import matplotlib.pyplot as plt
import FasterRCNN.quantization_utils as quant_utils
from FasterRCNN.FasterRCNN_MobilenetV2 import FasterRCNN_Generalized,MobileNetV2_Quantized
from FasterRCNN.staircase_dataset import StaircaseDataset
import FasterRCNN.pytorch_vision_utils.utils as utils
from FasterRCNN.pytorch_vision_utils.engine import evaluate

#------------------------ Argument parser ------------------------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
parser.add_argument('--mode',type= str, default='demo',help = 'Run either "demo" or "test" of the final model')
parser.add_argument('--img', type= str, default='STR_S3_80.JPG', help = 'Image name (taken from test data) to run the demo')
args = parser.parse_args()
#------------------------ Paths ------------------------
here = os.path.dirname(os.path.abspath(__file__))
trained_models_pth = os.path.join(here,'trained_models')
final_model_fname = 'FasterRCNN_Quantization_Aware.pth' 
final_model_path = os.path.join(trained_models_pth,final_model_fname)
test_root = os.path.join(here,'Dataset','test')
#------------------------ Load model ------------------------
cpu_device = cpu_device = torch.device("cpu:0")
model = quant_utils.load_torchscript_model(model_filepath=final_model_path,device=cpu_device)
cuda_false = False
#------------------------ Main code ------------------------
if args.mode == 'demo':
    print('Entering demo mode...')
    img_path = os.path.join(test_root,'images',args.img)
    if not os.path.isfile(img_path):
        print('Demo image not found')
        print('Refer to the README file for further instructions')
    
    annot_fname = args.img.split('.')[0]+'.xml'
    annot_path = os.path.join(test_root,'labels',annot_fname)
      
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    img.to(cpu_device)
    model.eval()
    output = model([img])
    if type(output)== tuple:
        output = output[1][0]
    
    if os.path.isfile(annot_path):
        annot_exist = True
        annot_path = open(annot_path,'r')
        xml_content = annot_path.read()
        xml_dict = xmltodict.parse(xml_content)
        annot_dict = dict(xml_dict['annotation']['object']['bndbox'])
        xmin = int(annot_dict['xmin'])
        xmax = int(annot_dict['xmax'])
        ymin = int(annot_dict['ymin'])
        ymax = int(annot_dict['ymax'])

        annot = [(xmin, ymin), (xmax, ymax)]
    
    pred = output['boxes'][0].detach().numpy()
    pred = [(pred[0],pred[1]),(pred[2],pred[3])]
    
    img_plot = cv2.imread(img_path)
    img_plot = cv2.rectangle(img_plot,pred[0],pred[1],(0,0,255),2)
    
    if annot_exist:
        img_plot = img_plot = cv2.rectangle(img_plot,annot[0],annot[1],(0,255,0),5)
    
    demo_results = os.path.join(here,'Results','predictions.jpg')
    #utils.mkdir(demo_results)
    img_plot = cv2.cvtColor(img_plot,cv2.COLOR_BGR2RGB)
    plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    plt.imshow(img_plot)
    plt.savefig(demo_results)

    print('Saved demo results in: {} '.format(demo_results))
    print('Annotations appear in green and predictions in red')

elif args.mode == 'test':
    print('Entering test mode...')
    # Test data loader
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
