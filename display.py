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
import pickle


def name(index):
    if (index == 0):
        return "outline_lateral"
    elif (index == 1):
        return "heart"
    elif (index == 2):
        return "yolk"
    elif (index == 3):
        return "ov"
    elif (index == 4):
        return "eyes_dorsal"
    else:
        return "outline_dorsal"
    return

model_seg = torch.load('static/weight/weights.pt' , map_location = torch.device('cpu'))
model_bool = torch.load('static/weight/weights_bool.pt' , map_location = torch.device('cpu'))
device = 'cpu'

def print_mask(img_path,well_path, type):
    matplotlib.use('agg')
    img = Image.open(img_path)
    data_transforms = transforms.Compose([transforms.ToTensor()])
    data_img = data_transforms(img)
    inputs = data_img.unsqueeze(0).to(device)
    outputs = model_seg(inputs)
    cmaps = ['Oranges', 'Purples', 'Blues', 'Greens', 'Greys', 'Reds']
    if type == "dorsal":
        plt.imshow(torch.split(data_img.unsqueeze(0),1,0)[0].squeeze().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(well_path +'/'+'dorsal.png', bbox_inches='tight')
        plt.clf()
        for i in [4,5]:
            mask = torch.split(outputs['out'].cpu(),1,0)[0].squeeze()[i].detach().numpy()
            mask = np.ma.masked_where(mask < -1, mask)
            plt.imshow(mask, cmaps[i], alpha = 0.7)
            plt.axis('off')
            part = name(i)
            plt.savefig(well_path +'/'+ part +'_out.png', bbox_inches='tight', transparent=True)
            plt.clf()
    else:
        plt.imshow(torch.split(data_img.unsqueeze(0),1,0)[0].squeeze().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(well_path +'/'+'lateral.png', bbox_inches='tight')
        plt.clf()
        for i in range(4):
            mask = torch.split(outputs['out'].cpu(),1,0)[0].squeeze()[i].detach().numpy()
            mask = np.ma.masked_where(mask < -1, mask)
            plt.imshow(mask, cmaps[i], alpha = 0.7)
            plt.axis('off')
            part = name(i)
            plt.savefig(well_path +'/'+ part +'_out.png', bbox_inches='tight', transparent=True)
            plt.clf()


def boolean(diccionario, well_name, image_lat, image_dor):
    input = input_boolean_net(image_lat,image_dor, "rgb", transforms.Compose([transforms.ToTensor()]))
    input = input.unsqueeze(0).to(device)
    outputs = model_bool(input)
    diccionario[well_name] = torch.sigmoid(outputs).squeeze().tolist()


def input_boolean_net(lat_image_path,dor_image_path,image_color_mode,transforms):
    with open(lat_image_path, "rb") as image_lat_file:
    #    Read the image and save it in the dictionary
        image_lat = Image.open(image_lat_file)
        if image_color_mode == "rgb":
            image_lat = image_lat.convert("RGB")
        elif image_color_mode == "grayscale":
            image_lat = image_lat.convert("L")
        image_lat = transforms(image_lat)

    with open(dor_image_path, "rb") as image_dor_file:
        '''
        Read the image and save it in the dictionary
        '''
        image_dor = Image.open(image_dor_file)
        if image_color_mode == "rgb":
            image = image_dor.convert("RGB")
        elif image_color_mode == "grayscale":
            image = image_dor.convert("L")
        image_dor = transforms(image_dor)

    return torch.cat((image_dor,image_lat),dim = 1)


def plate(plate_name, upload_folder):
    for plate_name in os.listdir(upload_folder):
        plate_path = upload_folder + "/" + plate_name
        tree = ET.parse(plate_path + "/" + plate_name + ".xml")
        # The root is the plate
        plate = tree.getroot()
        booleans = {}
        # Every child of the plate is a well
        for well in tqdm(plate):
            # If show2user is 0 we can skip the well
            if int(well.attrib['show2user']):
                # If we iterate over the well we obtain the boolean features and other stuff
                well_name = well.attrib['well_folder']
                well_path = plate_path + "/" + well_name

                # Images' paths
                dorsal_img_path = well_path + "/" + well.attrib['dorsal_image']
                lateral_img_path = well_path + "/" + well.attrib['lateral_image']
                image_name = plate_name + "_" + well_name
                #This list will contain pairs of (path, image) that will be written at the end if there are no errors.
                try:
                    print_mask(lateral_img_path,well_path, "lateral")
                    print_mask(dorsal_img_path,well_path, "dorsal")
                    boolean(booleans, well_name, lateral_img_path, dorsal_img_path)
                except:
                    continue

    with open('static/dict/booleans.pckl', 'wb') as handle:
        pickle.dump(booleans, handle, protocol=pickle.HIGHEST_PROTOCOL)
