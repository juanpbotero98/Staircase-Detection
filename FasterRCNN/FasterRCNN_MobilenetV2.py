import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from . import mobilenetv2 as mobilenet
import torch.nn as nn
import torch

def FasterRCNN_MobileNetV2():
    # Load backbone model 
    backbone = mobilenet.mobilenet_v2(pretrained=True).features
    # Define number of outputs
    backbone.out_channels = 1280
    # Define anchor generator
    anchor_generator = AnchorGenerator(sizes=((32,64,128,256,512),),aspect_ratios=((0.5,1.0,2.0),))
    # Define regions of interest (roi) cropping
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    # Define the FasterRCNN model
    model = FasterRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler,)
    return model

def FasterRCNN_Generalized(backbone):
    # Define number of outputs
    backbone.out_channels = 1280
    # Define anchor generator
    anchor_generator = AnchorGenerator(sizes=((32,64,128,256,512),),aspect_ratios=((0.5,1.0,2.0),))
    # Define regions of interest (roi) cropping
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    # Define the FasterRCNN model
    model = FasterRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler,)
    return model

class MobileNetV2_Quantized(nn.Module):
    def __init__(self, model_fp32):
        
        super(MobileNetV2_Quantized, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

