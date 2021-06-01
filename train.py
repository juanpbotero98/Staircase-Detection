import FasterRCNN.pytorch_vision_utils.utils as utils
from FasterRCNN.pytorch_vision_utils.engine import train_one_epoch, evaluate
from FasterRCNN.FasterRCNN_MobilenetV2 import FasterRCNN_MobileNetV2,FasterRCNN_Generalized,MobileNetV2_Quantized
from FasterRCNN.staircase_dataset import StaircaseDataset
from FasterRCNN import mobilenetv2 as mobilenet
import FasterRCNN.quantization_utils as quant_utils
import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms


#------------------------ Argument parser ------------------------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
parser.add_argument('-g', '--GPU_ID', type=int, default = 1, help='gpu id')
parser.add_argument('--num_epochs', type=int, default=10, help = 'Number of epochs to train the model')
parser.add_argument('--resume',type = int,default = 0,help = 'To resume training <1>')
parser.add_argument('--ckpt_path',type = str, default = os.path.join('trained_models','FasterRCNN_MobileNetV2_Quantized.pth'),
                    help = 'Path of the checkpoint file to resume training')
parser.add_argument('--resume_epoch',type = int,default = 0,help = 'Number of epochs in which the ckpt has been trained')
parser.add_argument('--quant',type = int,default = 1, help = 'To train quantized model <1>')
args = parser.parse_args()
here = os.path.dirname(os.path.abspath(__file__))

# Set environment variables
cuda = torch.cuda.is_available()
# num_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',').__len__()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU_ID)#','.join(f'{i}' for i in range(num_gpus))

torch.backends.cuda.cufft_plan_cache.clear()

torch.manual_seed(1337)
if cuda:
    torch.cuda.manual_seed(1337)

#------------------------ Data loaders ---------------------------

train_root = os.path.join(here,'Dataset','train')
test_root = os.path.join(here,'Dataset','test')
# std = [0.229, 0.224, 0.225]
# mean = [0.485, 0.456, 0.406]
transform = transforms.Compose([
                    transforms.ToTensor()
                ])

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
# Train data loader
train_loader = torch.utils.data.DataLoader(StaircaseDataset(train_root,transform,target_transform=None),batch_size=2,
                                            shuffle=True,**kwargs,collate_fn=utils.collate_fn)
# Test data loader
test_loader = torch.utils.data.DataLoader(StaircaseDataset(test_root,transform,target_transform=None),batch_size=2,
                                            shuffle=True,**kwargs,collate_fn=utils.collate_fn)

#------------------------ Model training -------------------------
# Define the model using helper model function
if not args.resume:
    if args.quant:
        backbone = mobilenet.mobilenet_v2(pretrained=True).features
        quant_backbone = MobileNetV2_Quantized(backbone)
        model = FasterRCNN_Generalized(quant_backbone)
        quant_utils.fuse_model(model.backbone)
        quant_config = torch.quantization.get_default_qconfig("fbgemm")
        model.backbone.qconfig = quant_config
        torch.quantization.prepare_qat(model, inplace=True)
        
    else:
        model = FasterRCNN_MobileNetV2()

else:
    model_path = os.path.join(here,args.ckpt_path)
    model = torch.load(model_path)

# Move the model to the desired GPU or CPU
cuda_device = torch.device('cuda:'+str(args.GPU_ID)) if cuda else torch.device('cpu')
cpu_device = torch.device("cpu:0")

# Create output directory
if args.quant:
    args.out = os.path.join(here,'QAtraining_logs')
    device = cpu_device
else: 
    args.out = os.path.join(here,'training_logs')
    device = cuda_device
utils.mkdir(args.out)

# Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)

# Add learning rate scheduker which decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

# Start training

for epoch in range(args.num_epochs):
    # Write current state to txt file
    f = open(os.path.join(args.out,'training_state.txt'),'a+')
    f.write("Running epoch {} ... \r\n".format(epoch+args.resume_epoch))
    f.close()
    # Move model to GPU or CPU     
    model = model.to(cuda_device)
    # Train for one epoch, printing every 10 iterations
    train_one_epoch(model,optimizer,train_loader,cuda,cuda_device,epoch,print_freq=50)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    #cuda_false = False

    if not args.quant:
        ev=evaluate(model, test_loader,cuda,cuda_device)
        out_pth = os.path.join(args.out,'checkpoint_epoch-{}.pth'.format(epoch+args.resume_epoch))
        # save model checkpoint
        torch.save(model,out_pth)
        # Remove previos checkpoint if it exists
        rm_path = os.path.join(args.out,'checkpoint_epoch-{}.pth'.format(epoch+args.resume_epoch-1))
        if os.path.exists(rm_path):
            os.remove(rm_path)
    else:
        if epoch > 3:
            # Freeze quantizer parameters
            model.apply(torch.quantization.disable_observer)
        if epoch > 2:
            # Freeze batch norm mean and variance estimates
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
        model.to(cpu_device)
        quant_model = torch.quantization.convert(model.eval(), inplace=False)
        quant_model.eval()
        quant_model.to(device)
        evaluate(quant_model, test_loader,cuda,device)
    
    

if args.quant:
    quant_model_filename = 'FasterRCNN_Quantization_Aware.pth'
    quant_utils.save_torchscript_model(model=quant_model, model_dir=args.out, model_filename=quant_model_filename)

print('Finished training model')

    

