import os
import cv2
import csv
import math
import enum
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from tqdm import tqdm
from PIL import Image
from copy import copy, deepcopy
from torchvision import transforms
from read_roi import read_roi_file
from roifile import ImagejRoi

'''
Loads an image from a path, applies transforms to it and returns it as a tensor
'''
def load_image(image_path,image_color_mode,transforms):
    with open(image_path, "rb") as image_file:
        '''
        Read the image and save it in the dictionary
        '''
        image = Image.open(image_file)
        if image_color_mode == "rgb":
            image = image.convert("RGB")
        elif image_color_mode == "grayscale":
            image = image.convert("L")
        image = transforms(image)
    return image

def leftmost_point(mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                return i,j
    return -1,-1

def rightmost_point(mask):
    for i in range(mask.shape[0]-1,-1,-1):
        for j in range(mask.shape[1]-1,-1,-1):
            if mask[i][j] != 0:
                return i,j
    return -1,-1

def upmost_point(mask):
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if mask[j][i] != 0:
                return j,i
    return -1,-1

def bottommost_point(mask):
    for i in range(mask.shape[1]-1,-1,-1):
        for j in range(mask.shape[0]-1,-1,-1):
            if mask[j][i] != 0:
                return j,i
    return -1,-1

def get_mask_extremes(mask):
    lx,ly = leftmost_point(mask)
    rx,ry = rightmost_point(mask)
    return lx,rx

'''
Given the outline_lat and eyes_lat masks, crops the outline_lat mask in half
and gets the half where the eyes fall.
'''
def head_mask(outline_lat_mask,eyes_dor_mask):
    o = outline_lat_mask.squeeze().numpy().transpose(1,0)
    e = eyes_dor_mask.squeeze().numpy().transpose(1,0)
    left_o,right_o = get_mask_extremes(o)
    left_e,right_e = get_mask_extremes(e)
    middle_o = (left_o + right_o)//2
    '''
    Get the limits of the head
    '''
    if left_e < middle_o:
        left_h = left_o
        right_h = middle_o
        right_h = (right_h + left_h)//2
    else:
        left_h = middle_o
        right_h = right_o
        left_h = (right_h + left_h)//2
    aux = np.zeros(o.shape,np.uint8)
    aux[left_h:right_h+1] = o[left_h:right_h+1]*255
    return aux.transpose(1,0)

'''
Given the outline_dors and eyes_lat masks, crops the outline_lat mask in half
and gets the half where the eyes fall.
'''
def fins_mask(dor_mask, eyes_dor_mask):
    o = dor_mask.squeeze().numpy().transpose(1,0)
    e = eyes_dor_mask.squeeze().numpy().transpose(1,0)
    left_o,right_o = Tox.get_mask_extremes(o)
    left_e,right_e = Tox.get_mask_extremes(e)
    middle_o = (left_o + right_o)//2
    '''
    Get the limits of the fins
    '''
    aux = np.zeros(o.shape,np.uint8)
    aux_ones = np.ones(o.shape,np.uint8)
    if right_e < middle_o:
        aux[right_e:middle_o] = aux_ones[right_e:middle_o]*255
    else:
        aux[middle_o:left_e] = aux_ones[middle_o:left_e]*255


    return aux.transpose(1,0)


'''
Given the outline_lat and eyes_lat masks, crops the outline_lat mask in half
and gets the half where the eyes fall.
'''
def head_and_fins_mask(outline_lat_mask,eyes_dor_mask):
    o = outline_lat_mask.squeeze().numpy().transpose(1,0)
    e = eyes_dor_mask.squeeze().numpy().transpose(1,0)
    left_o,right_o = get_mask_extremes(o)
    left_e,right_e = get_mask_extremes(e)
    middle_o = (left_o + right_o)//2
    '''
    Get the limits of the head
    '''
    if left_e < middle_o:
        left_h = left_o
        right_h = middle_o
        right_h = (right_h + left_h)//2
    else:
        left_h = middle_o
        right_h = right_o
        left_h = (right_h + left_h)//2
    aux = np.zeros(o.shape,np.uint8)
    aux[left_h:right_h+1] = o[left_h:right_h+1]*255
    '''
    Get the limits of the fins
    '''
    aux_fins = np.zeros(o.shape,np.uint8)
    aux_fins_ones = np.ones(o.shape,np.uint8)
    if right_e < middle_o:
        aux_fins[right_e:middle_o] = aux_fins_ones[right_e:middle_o]*255
    else:
        aux_fins[middle_o:left_e] = aux_fins_ones[middle_o:left_e]*255
    return aux.transpose(1,0), aux_fins.transpose(1,0)

'''
Returns only the part of the image where the mask is equal to 1.
'''
def inside_mask(ref_im, mask):
    x_max = ref_im.size()[1]
    y_max = ref_im.size()[2]
    ret_ima = torch.zeros(3, x_max, y_max)
    for i in range (0, x_max):
        for j in range (0, y_max):
            for c in range(3):
                ret_ima[c][i][j] = min(float(ref_im[c][i][j]), float(mask[0][i][j]))
    return ret_ima

'''
Applies a threshold to the output of the 2D matrix representing the mask and
saves it in the given path.
'''
def save_mask_prediction(mask, mask_path, threshold = 0.5):
    _, mask = cv2.threshold(mask,threshold,255,cv2.THRESH_BINARY)
    #hist = cv.calcHist([mask],[0],None,[256],[0,256])
    cv2.imwrite(mask_path,mask)

def ellipse_to_pol(roi):
    pts = []
    p1 = np.array([roi['ex1'],roi['ey1']])
    p2 = np.array([roi['ex2'],roi['ey2']])
    a = np.linalg.norm(p1-p2)/2
    c = (p1+p2)/2
    b = a*roi['aspect_ratio']
    vec = np.arange(roi['ex1'],roi['ex2'],1)
    vec2 = np.arange(roi['ex2'],roi['ex1'],-1)
    for i in vec:
        pts.append([i,b*math.sqrt(1-((i-c[0])**2/a**2))+c[1]])
    for i in vec2:
        pts.append([i,-b*math.sqrt(1-((i-c[0])**2/a**2))+c[1]])
    return np.array(pts, 'int32')

# Given the path of the image and its corresponding roi returns the image with a mask
# built from the roi
def obtain_mask(img,roi):
    if roi['type'] in ['polygon', 'traced']:
        pts = zip(roi['x'],roi['y'])
        pts2 = np.array(list(pts), 'int32')
        #Important to put the brackets []!!!!
        cv2.fillPoly( img , [pts2], (255))
    elif roi['type'] == 'freehand':
        #Important to put the brackets []!!!!
        cv2.fillPoly( img , [ellipse_to_pol(roi)], (255))
    return img

'''
input:
  roi_paths: list of absolute paths of the roi files
  mask_name: list of names of the masks to access them in the roi dictionaries
  root_data_folder: parent folder of all the output mask folders
  mask_folder: name of the output folder
  im_type: "dor" or "lat"
  image_name: name of the image to match with ("plate_name"_"well_name")
  shape: (width, height) of the output mask image
'''
def read_roi_and_get_mask(roi_paths,mask_names,root_data_folder,mask_folder,im_type,image_name,shape):
    #Create Black image to put the masks on:
    mask_img = np.zeros(shape, np.uint8)
    for i,(roi_path,mask_name) in enumerate(zip(roi_paths, mask_names)):
        #Get the roi
        roi = read_roi_file(roi_path)[mask_name]
        #Create the mask
        mask_img = obtain_mask(mask_img,roi)
        #Define the path to be written to
        mask_path = root_data_folder + "/" + mask_folder + "/" + image_name + "_" + im_type + ".jpg"
    return mask_path, mask_img

'''
Given a root path, a list of names for the masks folders and a name for
the input images folder, creates all the folders if they are not yet created
'''
def create_directories(output_root,mask_folders,images_folder):
    if not os.path.exists(output_root):
        os.system("mkdir " + output_root)
    if not os.path.exists(output_root + "/" + images_folder):
        os.system("mkdir " + output_root + "/" + images_folder)
    for folder in mask_folders:
        if not os.path.exists(output_root + "/" + folder):
            os.system("mkdir " + output_root + "/" + folder)

'''
given an image name in the form: plate_name + _ + well_name + _ + type + .jpg,
return plate_name, well_name

example:
    image_name = 20190902_1046_CS_R1_Well_left_A01_dor.jpg
    return: 20190902_1046_CS_R1, Well_left_A01
'''
def parse_image_name(image_name,has_tail):
    s = image_name.split('Well')
    #get the first element and remove the "_" at the end
    plate_name = s[0][:-1]
    well_name = "Well" + s[1]
    if has_tail: well_name = well_name[:-8]
    return plate_name, well_name

def data_generation_pipeline(raw_data_paths, output_folder, masks_names):
    # Create the folders if they are not created yet
    create_directories(output_folder, masks_names.keys(), 'Images')
    # Initialize the different lists of image_names
    complete_fishes = []
    complete_bools = []
    all_fishes = []
    #init stats csv file
    feno_names = ['bodycurvature', 'yolkedema', 'necrosis', 'tailbending', 'notochorddefects', 'craniofacialedema', 'finabsence', 'scoliosis', 'snoutjawdefects', 'otolithsdefects']
    fieldnames = ['experiment','plate','well','image_name','compound','exposure','dose','Image_lat','Image_dor'] + list(masks_names.keys()) + feno_names
    with open(os.path.join(output_folder, 'stats.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    # Iterate over every plate
    for raw_data_path in raw_data_paths:
        experiment_name = raw_data_path.split('/')[-1]
        print(f'Generating: {experiment_name}\'s Data')
        for plate_name in tqdm(os.listdir(raw_data_path)):
            plate_path = raw_data_path + "/" + plate_name
            #Some plates don't have an XML:
            try:
                tree = ET.parse(plate_path + "/" + plate_name + ".xml")
            except:
                print(f'{plate_name} has no XML')
                continue
            # The root is the plate
            plate = tree.getroot()

            # Every child of the plate is a well
            for well in plate:
                # If show2user is 0 we can skip the well
                if int(well.attrib['show2user']):
                    stats = {f:None for f in fieldnames}
                    # If we iterate over the well we obtain the boolean features and other stuff
                    well_name = well.attrib['well_folder']
                    well_path = plate_path + "/" + well_name

                    # Images' paths
                    dorsal_img_path = well_path + "/" + well.attrib['dorsal_image']
                    lateral_img_path = well_path + "/" + well.attrib['lateral_image']
                    image_name = plate_name + "_" + well_name
                    # This list will contain pairs of (path, image) that will be written at the end if there are no errors.
                    outputs = []
                    stats['experiment'] = experiment_name
                    stats['plate'] = plate_name
                    stats['well'] = well_name
                    stats['image_name'] = image_name
                    stats['compound'] = well.attrib['compound']
                    stats['exposure'] = well.attrib['exposure']
                    stats['dose'] = well.attrib['dose']
                    try:
                        im_lat = cv2.imread(lateral_img_path, 1)
                        outputs.append((output_folder + "/Images/" + image_name + "_lat.jpg", im_lat))
                        all_fishes.append(image_name + '_lat.jpg')
                        stats['Image_lat'] = True
                    except:
                        stats['Image_lat'] = False
                    try:
                        im_dor = cv2.imread(dorsal_img_path, 1)
                        outputs.append((output_folder + "/Images/" + image_name + "_dor.jpg", im_dor))
                        stats['Image_dor'] = True
                        all_fishes.append(image_name + '_dor.jpg')
                    except:
                        stats['Image_dor'] = False

                    roi_paths = []

                    for masks in masks_names.values():
                        roi_names = []
                        for mask in masks:
                            roi_names.append(well_path + '/' + mask + '.roi')
                        roi_paths.append(roi_names)

                    # We get the shape of the original image to create the mask
                    height, width, _ = im_lat.shape
                    mask_shape = (height, width, 1) # 1 because is greyscale and not RGB
                    # Generate all the masks
                    for i, (mask_folder, masks) in enumerate(masks_names.items()):
                        try:
                            mask_path, mask = read_roi_and_get_mask(roi_paths[i],
                                                                    masks,
                                                                    output_folder,
                                                                    mask_folder,
                                                                    mask_folder[-3:],
                                                                    image_name,
                                                                    mask_shape)
                            outputs.append((mask_path, mask))
                            stats[mask_folder] = True
                        except:
                            stats[mask_folder] = False

                    #Get boolean phenotypes
                    translate_feno_nomenclature = {
                    'False' : 0,
                    'True' : 1,
                    'ND' : None
                    }
                    complete_bool = True
                    for feno in well:
                        if 'value' in feno.attrib.keys() and feno.tag in feno_names:
                            stats[feno.tag] = translate_feno_nomenclature[feno.attrib['value']]
                            if stats[feno.tag] is None:
                                complete_bool = False
                        elif feno.tag in fieldnames:
                            stats[feno.tag] = None
                            complete_bool = False
                    #check that all phenotypes have been covered
                    for feno in feno_names:
                        if stats[feno] is None:
                            complete_bool = False

                    # Write in log.csv batchsummary
                    with open(os.path.join(output_folder, 'stats.csv'), 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(stats)

                    #Update complete fishes in the image classification sense
                    if complete_bool:
                        complete_bools.append(image_name)
                    #Update complete fishes in the image segmentation sense
                    if all([stats['Image_lat'],stats['outline_lat'],stats['heart_lat'],stats['yolk_lat'],stats['ov_lat']]):
                        complete_fishes.append(image_name + '_lat.jpg')
                    if all([stats['Image_dor'],stats['outline_dor'],stats['eyes_dor']]):
                        complete_fishes.append(image_name + '_dor.jpg')

                    #If we are here, everything went fine, so we can write the images
                    for image_path, image in outputs:
                        cv2.imwrite(image_path, image)
    with open(os.path.join(output_folder, 'complete_fishes.pkl'), "wb") as f:
        pickle.dump(complete_fishes, f)
    with open(os.path.join(output_folder, 'complete_bools.pkl'), "wb") as f:
        pickle.dump(complete_bools, f)
    with open(os.path.join(output_folder, 'all_fishes.pkl'), "wb") as f:
        pickle.dump(all_fishes, f)

'''
Receives a dataloader and returns the number of possitive and negative samples
for all the boolean fenotypes.
output:
    dict = {'feno_name': (n_pos,n_neg), ...}
'''
def get_stats_bool_fenotypes(dataset,stats_path,feno_names):
    results = {feno_name:[0,0] for feno_name in feno_names}
    stats = pd.read_csv(stats_path)
    image_names = getattr(dataset,'image_names')
    for feno_name in results:
        fen = stats[stats['image_name'].isin(image_names)][feno_name]
        pos_examples = sum(fen)
        total_examples = int(fen.shape[0])
        results[feno_name][0] += pos_examples
        results[feno_name][1] += (total_examples - pos_examples)
    return results

'''
Given a list of image names, filters it by only keeping the fish_names with
all the boolean phenotypes in feno_names
'''
def filter_by_bool(images_list,feno_names,stats_path):
    stats = pd.read_csv(stats_path)
    image_set = set([i[:-8] for i in images_list])
    stats = stats[stats['image_name'].isin(image_set)]
    stats = stats[feno_names + ['image_name']]
    complete_bools = stats.dropna()
    print(len(stats),len(complete_bools))
    return list(complete_bools['image_name'])


'''
Given an array of images, their names in the Well format, a batch size, an orientation list
conformed by "lat" or "dor" and the segmentation model,return a dictionary where
the key is the name of the image and the value is another dictionary with a key and
value corresponding to each mask extracted from the image.
'''
def predict_terato_masks_deep(images,image_names,mask_names,batch_size, orientation_list,model,device):
    im = torch.cat([i.unsqueeze(0) for i in images],dim = 0)
    input_tensor = im.to(device)
    with torch.set_grad_enabled(False):
        output_tensor = torch.sigmoid(model(input_tensor)['out'])#.to(device)

        # Thresholds for each mask
        thresholds = [0.22, 0.2, 0.5, 0.01, 0.1, 0.1]
        masks_dict = {name:{mask:None for mask in mask_names} for name in image_names}

        for i in range(batch_size):
            if orientation_list[i] == "lat":
                outline_lat = output_tensor[i][0].cpu().detach().numpy()
                heart_lat = output_tensor[i][1].cpu().detach().numpy()
                yolk_lat = output_tensor[i][2].cpu().detach().numpy()
                ov_lat = output_tensor[i][3].cpu().detach().numpy()
                # Threshold and add to dictionary
                _, masks_dict[image_names[i]]["outline_lat"] =cv2.threshold(outline_lat,thresholds[0],1,cv2.THRESH_BINARY)
                _, masks_dict[image_names[i]]["heart_lat"] =cv2.threshold(heart_lat,thresholds[1],1,cv2.THRESH_BINARY)
                _, masks_dict[image_names[i]]["yolk_lat"] =cv2.threshold(yolk_lat,thresholds[2],1,cv2.THRESH_BINARY)
                _, masks_dict[image_names[i]]["ov_lat"] =cv2.threshold(ov_lat,thresholds[3],1,cv2.THRESH_BINARY)
            else:
                eyes_dor = output_tensor[i][4].cpu().detach().numpy()
                outline_dor = output_tensor[i][5].cpu().detach().numpy()
                # Threshold and add to dictionary
                _, masks_dict[image_names[i]]["eyes_dor"] =cv2.threshold(eyes_dor,thresholds[4],1,cv2.THRESH_BINARY)
                _, masks_dict[image_names[i]]["outline_dor"] =cv2.threshold(outline_dor,thresholds[5],1,cv2.THRESH_BINARY)



    return masks_dict

def split_eye_masks(mask):
    lx,ly = upmost_point(mask)
    rx,ry = bottommost_point(mask)

    middle = (rx+lx)//2

    up = np.zeros(mask.shape,np.uint8)
    down = np.zeros(mask.shape,np.uint8)
    # Generate up mask
    for i in range(middle):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                up[i][j] = mask[i][j]
    # Generate down mask
    for i in range(middle, mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                down[i][j] = mask[i][j]

    return up, down

class ROI_TYPE(enum.IntEnum):
    POLYGON = 0
    RECT = 1
    OVAL = 2
    LINE = 3
    FREELINE = 4
    POLYLINE = 5
    NOROI = 6
    FREEHAND = 7
    TRACED = 8
    ANGLE = 9
    POINT = 10

# Mask is the loaded mask, tensor
# Well path where to save the roi
# Mask_name is outline_dor, ov_dor,...
def generate_and_save_roi(mask,well_path,mask_name):
    mask2 = np.array(mask,dtype=np.uint8)
    if mask_name != "eyes_dor":

        contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        contours = np.squeeze(np.array(contours))


    if mask_name == "outline_lat":
        roi_path = os.path.join(well_path, "fishoutline_lateral.roi")
    elif mask_name == "heart_lat":
        roi_path = os.path.join(well_path, "heart_lateral.roi")
    elif mask_name == "yolk_lat":
        roi_path = os.path.join(well_path, "yolk_lateral.roi")
    elif mask_name == "ov_lat":
        roi_path = os.path.join(well_path, "ov_lateral.roi")
    elif mask_name == "eyes_dor": # Split the eyes mask between up and down eye
        # Split masks
        up_eye_mask, down_eye_mask = split_eye_masks(mask2)

        # Extract contours
        roi_up, _ = cv2.findContours(up_eye_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        roi_up = np.squeeze(np.array(roi_up))
        if len(roi_up) > 10:
            roi_up= ImagejRoi.frompoints(roi_up)
            roi_path_up = os.path.join(well_path, "eye_up_dorsal.roi")
            roi_up.tofile(roi_path_up)
        else:
            print("Error in mask:", mask_name)

        roi_down, _ = cv2.findContours(down_eye_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        roi_down = np.squeeze(np.array(roi_down))
        if len(roi_down) > 10:
            roi_down= ImagejRoi.frompoints(roi_down)
            roi_down.roitype = ROI_TYPE.POLYGON
            roi_path_down = os.path.join(well_path, "eye_down_dorsal.roi")
            roi_down.tofile(roi_path_down)
        else:
            print("Error in mask:", mask_name)

        return None
    elif mask_name == "outline_dor":
        roi_path = os.path.join(well_path, "fishoutline_dorsal.roi")
    else:
        roi_path = os.path.join(well_path, "bad_mask_name.roi")
        print("Incorrect mask_name, saving to", roi_path)

    if len(contours) > 10:
        roi= ImagejRoi.frompoints(contours)
        roi.tofile(roi_path) # Save roi to file
    else:
        print("Error in mask:", mask_name)

def predict_terato_bools_deep(images,image_names,feno_names,model,device):
    fenos = {im:{f:None for f in feno_names} for im in image_names}
    input = torch.cat([i.unsqueeze(0) for i in images], dim=0).to(device)
    model.eval()
    with torch.set_grad_enabled(False):
        out = torch.sigmoid(model(input)).cpu()
        for i,im in enumerate(image_names):
            for j,feno in enumerate(feno_names):
                fenos[im][feno] = float(out[i][j])
    return fenos

def area(im):
    return np.sum(im)

def length(im):
    lx, rx = get_mask_extremes(im)
    return rx-lx

def generate_and_save_predictions(plate_path,batch_size,model_path_seg,model_path_bools,mask_names,feno_names,device,path_dataframe = None):
    if path_dataframe is None:
        path_dataframe = plate_path + '/stats.csv'
    column_names_area = ['area_out_lat','area_heart', 'area_yolk', 'area_ov', 'area_eyes', 'area_out_dor']
    column_names_length = ['length_out_lat','length_heart','length_yolk','length_ov','length_eyes','length_out_dor']

    df = pd.DataFrame(columns=['plate', 'well_folder', 'compound', 'dose', 'dead120', 'dead24'])

    trans = transforms.Compose([transforms.ToTensor()])
    model_seg = torch.load(f'./{model_path_seg}',map_location=device)
    model_bools = torch.load(f'./{model_path_bools}',map_location=device)
    plate_name = plate_path.split('/')[-1]
    xml_path = plate_path + "/" + plate_name + ".xml"
    tree = ET.parse(xml_path)
    plate = tree.getroot()
    data_frame = {columns_name:[] for columns_name in column_names_area + column_names_length + feno_names}

    batch = 0
    types = []
    image_names = []
    image_names_bool = []
    images = []
    images_bool = []
    well_list = []
    well_paths = []
    # Every child of the plate is a well
    for well in tqdm(plate):
        well_name = 'Well_' + well.attrib['name']
        df = df.append({'plate': plate_name, 'well_folder': well_name, 'compound':
                        well.attrib['compound'], 'dose': float(well.attrib['dose']), 'dead120':
                        int(well.attrib['dead120']), 'dead24': int(well.attrib['dead24'])}, ignore_index = True)
        # If show2user is 0 we can skip the well
        if int(well.attrib['show2user']):
            batch += 2
            well_path = plate_path + "/" + well_name

            # Images' paths
            dorsal_img_path = well_path + "/" + well.attrib['dorsal_image']
            lateral_img_path = well_path + "/" + well.attrib['lateral_image']
            image_name = plate_name + "_" + well_name
            dorsal_image = load_image(dorsal_img_path,'rgb',trans)
            lateral_image = load_image(lateral_img_path,'rgb',trans)
            types = types + ['dor','lat']
            images = images + [dorsal_image,lateral_image]
            images_bool.append(torch.cat((lateral_image,dorsal_image),dim=1))
            image_names = image_names + [image_name,image_name]
            image_names_bool.append(image_name)
            well_list.append(well)
            well_paths.append(well_path)

            if batch == batch_size:
                dict = predict_terato_masks_deep(images,image_names,mask_names,batch_size,types,model_seg,device)
                for i, fish_name in enumerate(dict):
                    for j, mask in enumerate(dict[fish_name]):
                        data_frame[column_names_area[j]].append(area(dict[fish_name][mask]))
                        data_frame[column_names_area[j]].append(length(dict[fish_name][mask]))
                        generate_and_save_roi(dict[fish_name][mask],well_paths[i],mask)
                fenos = predict_terato_bools_deep(images_bool,image_names_bool,feno_names,model_bools,device)
                for w,im in zip(well_list,image_names_bool):
                    available_fenos = {f.tag:f for f in w}
                    for feno in feno_names:
                        data_frame[feno].append(fenos[im][feno] > 0.5)
                        if feno in available_fenos:
                            if 'finished' in available_fenos[feno].attrib:
                                if available_fenos[feno].attrib['value'] == 'ND':
                                    available_fenos[feno].set('probability',available_fenos[feno].attrib['value'])
                                else: available_fenos[feno].set('probability',-1)
                            else:
                                available_fenos[feno].set("value",str(fenos[im][feno] > 0.5))
                                available_fenos[feno].set("probability",str(fenos[im][feno]))
                        else:
                            child = ET.SubElement(w, feno)
                            child.set("value",str(fenos[im][feno] > 0.5))
                            child.set("probability",str(fenos[im][feno]))


                batch = 0
                types = []
                image_names = []
                image_names_bool = []
                images = []
                images_bool = []
                well_list = []
                well_paths = []
        else:
            for column_name in column_names_area + column_names_length + feno_names:
                data_frame[column_name].append("NA")

    for column in data_frame:
        df[column] = data_frame[column]
    df.to_csv(path_dataframe)
    tree = ET.ElementTree(plate)
    with open(xml_path, "wb") as fh:
        tree.write(fh)