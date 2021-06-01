import torch
import FasterRCNN.quantization_utils as quant_utils
from FasterRCNN.FasterRCNN_MobilenetV2 import FasterRCNN_MobileNetV2,FasterRCNN_Generalized,MobileNetV2_Quantized
from FasterRCNN import mobilenetv2 as mobilenet
import os 

# Paths
here = os.path.dirname(os.path.abspath(__file__))
out_pth = os.path.join(here,'trained_models')
model_fname = 'Quantizable_FasterRCNN_MobilenetV2.pth'
model_pth = os.path.join(out_pth,model_fname)
quant_model_fname = 'FasterRCNN_QAT_StateDict.pth'
quant_model_pth = os.path.join(out_pth,quant_model_fname)

# Enviroment variables
cpu_device = torch.device("cpu:0")
GPU_ID = 1
cuda_device = torch.device("cuda:"+str(GPU_ID))

# Floating point model
fp32_model = torch.load(model_pth,map_location=cuda_device)
input_size =(1,3,320,320) # batch,channels,width,height
inputs = torch.rand(size=input_size).to(cuda_device)

# Define quantized model
backbone = mobilenet.mobilenet_v2(pretrained=False).features
quant_backbone = MobileNetV2_Quantized(backbone)
quant_model = FasterRCNN_Generalized(quant_backbone)
quant_model.eval()
quant_config = torch.quantization.get_default_qconfig("fbgemm")
quant_model.backbone.qconfig = quant_config
torch.quantization.prepare(quant_model, inplace=True)
quant_model = torch.quantization.convert(quant_model, inplace= True)
#Load weights
#quant_statedict=torch.load(quant_model_pth)
quant_model.load_state_dict(torch.load(quant_model_pth))

# Evaluate FLOPS
bbone= fp32_model.backbone
bbone.to(cuda_device)
backbone_flops = quant_utils.get_flops(bbone,cuda_device,input_size)
fp32_flops = quant_utils.get_flops(fp32_model,cuda_device,input_size)
quant_flops = fp32_flops-backbone_flops
#quant_flops = quant_utils.get_flops(quant_model,cpu_device,input_size)

print('The flops of the fp32 model are: {}'.format(fp32_flops))
print('The flops of the int8 model are: {}'.format(quant_flops))

# Evaluate inference latency
fp32_latency_CPU = quant_utils.measure_inference_latency(fp32_model,cpu_device,False,input_size)
fp32_latency_GPU = quant_utils.measure_inference_latency(fp32_model,cuda_device,False,input_size)
quant_latency = quant_utils.measure_inference_latency(quant_model,cpu_device,True,input_size)

print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(
        fp32_latency_CPU* 1000))
print("FP32 GPU Inference Latency: {:.2f} ms / sample".format(
        fp32_latency_GPU* 1000))
print("Int8 CPU Inference Latency: {:.2f} ms / sample".format(
        quant_latency* 1000))