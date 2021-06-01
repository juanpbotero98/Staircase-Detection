import os
import numpy
import torch
import torch.utils.data
from PIL import Image
import xmltodict
import FasterRCNN.pytorch_vision_utils.transforms as T

class StaircaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.targetTransform = target_transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root,'images'))))
        self.labels = list(sorted(os.listdir(os.path.join(root,'labels'))))

    # Define function for data set transforms
    def get_transform():
        transforms = []
        # converts the image, a PIL image, into a PyTorch Tensor
        transforms.append(T.ToTensor())
        return T.Compose(transforms)

    def __getitem__(self, idx):
        # load images 
        img_path = os.path.join(self.root,'images', self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # load groundtruth
        gt_path = os.path.join(self.root,'labels',self.labels[idx])
        gt_path = open(gt_path,'r')
        xml_content = gt_path.read()
        xml_dict = xmltodict.parse(xml_content)
        # Load bounding boxes
        boxes = []
        # Check if there are multiple stairs in the image
        if type(xml_dict['annotation']['object']) == list:
            num_objs= len(xml_dict['annotation']['object'])
            for i in range(num_objs):
                annot_dict = dict(xml_dict['annotation']['object'][i]['bndbox'])
                xmin = int(annot_dict['xmin'])
                xmax = int(annot_dict['xmax'])
                ymin = int(annot_dict['ymin'])
                ymax = int(annot_dict['ymax'])
                boxes.append([xmin, ymin, xmax, ymax])

        
        # For single stairs
        else:
            num_objs = 1
            annot_dict = dict(xml_dict['annotation']['object']['bndbox'])
            xmin = int(annot_dict['xmin'])
            xmax = int(annot_dict['xmax'])
            ymin = int(annot_dict['ymin'])
            ymax = int(annot_dict['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])            

        # Convert list to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img = self.transform(img)

        if self.targetTransform is not None:
            target = self.targetTransform(target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)
    