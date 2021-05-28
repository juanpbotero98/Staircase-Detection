import os
import numpy
import torch
import torch.utils.data
from PIL import Image
import xmltodict

class StaircaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(root)))
        self.masks = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        # load images 
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # load groundtruth
        gt_path = os.path.join(self.root, self.masks[idx])
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
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # For single stairs
        annot_dict = dict(xml_dict['annotation']['object']['bndbox'])
        xmin = int(annot_dict['xmin'])
        xmax = int(annot_dict['xmax'])
        ymin = int(annot_dict['ymin'])
        ymax = int(annot_dict['ymax'])
        boxes.append([xmin, ymin, xmax, ymax])            

        # Convert list to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # there is only one class
        labels = torch.ones((1,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target