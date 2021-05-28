import os
from tqdm import tqdm
import shutil

test_pth = os.path.join('Dataset','test_labels')
test_list = os.listdir(test_pth)
train_pth = os.path.join('Dataset','train_labels')
train_list = os.listdir(train_pth)
imgs_pth = os.path.join('Dataset','Images')

for file in tqdm(train_list, desc = 'Copying train data to new folder'):
    file = file.split('.')[0] 
    old_pth = os.path.join(imgs_pth,file + '.jpg')
    new_pth = os.path.join(imgs_pth,'train',file + '.jpg')
    
    if not os.path.isfile(old_pth):
        old_pth = os.path.join(imgs_pth,file + '.JPG')
        new_pth = os.path.join(imgs_pth,'train',file + '.JPG')
    
    shutil.move(old_pth,new_pth)

for file in tqdm(test_list, desc = 'Copying test data to new folder'):
    file = file.split('.')[0] 
    old_pth = os.path.join(imgs_pth,file + '.jpg')
    new_pth = os.path.join(imgs_pth,'test',file + '.jpg')
    
    if not os.path.isfile(old_pth):
        old_pth = os.path.join(imgs_pth,file + '.JPG')
        new_pth = os.path.join(imgs_pth,'test',file + '.JPG')
    
    shutil.move(old_pth,new_pth)    
