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
import shutil
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


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



def print_mask(well_path, type):
    # Load the image and convert it to tensor
    matplotlib.use('Agg')
    img = Image.open(well_path + '/' + type)
    data_transforms = transforms.Compose([transforms.ToTensor()])
    data_img = data_transforms(img)
    # Define the colors
    cmaps = [ListedColormap(["#74CCB1"]), ListedColormap(["#CCCCFF"]), ListedColormap(["#F1948A"]),ListedColormap(["#F7DC6F"]),ListedColormap(["#85C1E9"]),ListedColormap(["#CD6155"])]
    # Add batch dimension to enter the NN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = data_img.unsqueeze(0).to(device)
    outputs = model_seg(input)

    plt.imshow(img)
    plt.axis('off')
    plt.savefig(well_path +'/'+ type.split('_')[0]+ '.png', bbox_inches='tight', transparent=True)
    plt.clf()


    ran = [4, 5] if type == 'dorsal_original.png' else range(0, 4)

    for i in ran:
        mask = torch.split(outputs['out'].cpu(),1,0)[0].squeeze()[i].detach().numpy()
        mask = np.ma.masked_where(mask < 0, mask)
        plt.imshow(mask, cmaps[i], alpha = 0.4)
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


def plate(upload_folder, plate_name):
    plate_path = upload_folder + "/" + plate_name
    tree = ET.parse(plate_path + "/" + plate_name + ".xml")
    # The root is the plate
    plate = tree.getroot()
    booleans = {}
    # Every child of the plate is a well
    for well in tqdm(plate):
        # If show2user is 0 we can skip the well
        well_name = well.attrib['well_folder']
        well_path = plate_path + "/" + well_name
        if int(well.attrib['show2user']):
            # If we iterate over the well we obtain the boolean features and other stuff

            # Images' paths
            dorsal_img_path = well_path + "/" + well.attrib['dorsal_image']
            lateral_img_path = well_path + "/" + well.attrib['lateral_image']
            list_photos = os.listdir(well_path)
            print(list_photos)
            for photo in list_photos:
                photo = well_path + '/' + photo
                if photo not in [dorsal_img_path, lateral_img_path]:
                    os.remove(photo)
                elif photo == dorsal_img_path:
                    os.rename(photo, well_path + '/' + 'dorsal_original.png')
                else:
                    os.rename(photo, well_path + '/' + 'lateral_original.png')

            #This list will contain pairs of (path, image) that will be written at the end if there are no errors.
            try:
                print_mask(well_path, "lateral_original.png")
                print_mask(well_path, "dorsal_original.png")
                boolean(booleans, well_name, well_path + "/" + "lateral_original.png",  well_path + "/" + "dorsal_original.png")
            except:
                shutil.rmtree(well_path, ignore_errors=True)
                continue
        else:
            shutil.rmtree(well_path, ignore_errors=True)
        try:
            os.remove(well_path+ '/' + "lateral_original.png", ignore_errors=True)
            os.remove(well_path+ '/' + "dorsal_original.png", ignore_errors=True)
        except:
            pass

    with open('static/dict/' + plate_name + '.pckl', 'wb') as handle:
        pickle.dump(booleans, handle, protocol=pickle.HIGHEST_PROTOCOL)
