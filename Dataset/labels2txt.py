import xmltodict
import os
from tqdm import tqdm

lbl_pth = os.path.join('Dataset','test_labels')
lbl_list = os.listdir(lbl_pth)
for file in tqdm(lbl_list, desc = 'Converting labels'):
    pth = os.path.join(lbl_pth,file)
    file_pth = open(pth,'r')
    xml_content = file_pth.read()
    xml_dict = xmltodict.parse(xml_content)
    txt_file = open(os.path.join('Dataset','groundtruths_txt',"{}.txt".format(file.split('.')[0])),"w")
    # for annot in xml_dict['annotation']['object']:
    #     annot_dict = dict(annot['bndbox'])
    if type(xml_dict['annotation']['object']) == list:
        for i in range(len(xml_dict['annotation']['object'])):
            annot_dict = dict(xml_dict['annotation']['object'][i]['bndbox'])
            txt_file.write('stairs {} {} {} {} \n'.format(int(annot_dict['xmin']),int(annot_dict['ymax']),int(annot_dict['xmax']),int(annot_dict['ymin'])))
    else:
        annot_dict = dict(xml_dict['annotation']['object']['bndbox'])
        txt_file.write('stairs {} {} {} {} '.format(int(annot_dict['xmin']),int(annot_dict['ymax']),int(annot_dict['xmax']),int(annot_dict['ymin'])))
