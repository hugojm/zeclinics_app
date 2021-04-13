import os
from PIL import Image, ImageTk
import os
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm


def name(index):
    if (index == 0):
        return "eyes_dorsal"
    elif (index == 1):
        return "outline_lateral"
    elif (index == 2):
        return "yolk"
    elif (index == 3):
        return "heart"
    elif (index == 4):
        return "outline_dorsal"
    else:
        return "ov"
    return

def print_mask(img_path, type):
    matplotlib.use('agg')
    device = 'cpu'
    img = Image.open(img_path)
    model = torch.load('static/weight/weights.pt' , map_location = torch.device('cpu'))
    data_transforms = transforms.Compose([transforms.ToTensor()])
    data_img = data_transforms(img)
    inputs = data_img.unsqueeze(0).to(device)
    outputs = model(inputs)
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    if type == "dorsal":
        for i in [0,4]:
            mask = torch.split(outputs['out'].cpu(),1,0)[0].squeeze()[i].detach().numpy()
            mask = np.ma.masked_where(mask < 0.1, mask)
            plt.imshow(torch.split(data_img.unsqueeze(0),1,0)[0].squeeze().permute(1, 2, 0), alpha=0.7)
            plt.imshow(mask, cmaps[i], alpha = 0.7)
            part = name(i)
            plt.savefig(img_path[:-4]+'_'+ part +'_out.png', bbox_inches='tight')
    else:
        for i in [1,2,3,5]:
            mask = torch.split(outputs['out'].cpu(),1,0)[0].squeeze()[i].detach().numpy()
            mask = np.ma.masked_where(mask < 0.1, mask)
            plt.imshow(torch.split(data_img.unsqueeze(0),1,0)[0].squeeze().permute(1, 2, 0), alpha=0.7)
            plt.imshow(mask, cmaps[i], alpha = 0.7)
            part = name(i)
            plt.savefig(img_path[:-4]+'_'+ part +'_out.png', bbox_inches='tight')


def plate(plate_name, upload_folder):
    for plate_name in tqdm(os.listdir(upload_folder)):
        plate_path = upload_folder + "/" + plate_name
        tree = ET.parse(plate_path + "/" + plate_name + ".xml")
        # The root is the plate
        plate = tree.getroot()

        # Every child of the plate is a well
        for well in plate:
            # If show2user is 0 we can skip the well
            if int(well.attrib['show2user']):
                # If we iterate over the well we obtain the boolean features and other stuff
                well_name = well.attrib['well_folder']
                well_path = plate_path + "/" + well_name

                # Images' paths
                dorsal_img_path = well_path + "/" + well.attrib['dorsal_image']
                lateral_img_path = well_path + "/" + well.attrib['lateral_image']
                image_name = plate_name + "_" + well_name
                # This list will contain pairs of (path, image) that will be written at the end if there are no errors.
                try:
                    print_mask(lateral_img_path, "lateral")
                    print_mask(dorsal_img_path, "dorsal")
                except:
                    continue
