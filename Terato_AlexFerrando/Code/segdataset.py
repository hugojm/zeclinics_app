from pathlib import Path
from typing import Any, Callable, Optional
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.datasets.vision import VisionDataset
from ETL_lib import parse_image_name
import os
import pickle
import torchvision.transforms.functional as F
import pandas as pd

def random_flip(dict, p):
    if torch.rand(1) < p:
        dict['image'] = F.hflip(dict['image'])
        dict['masks'] = F.hflip(dict['masks'])
    if torch.rand(1) < p:
        dict['image'] = F.vflip(dict['image'])
        dict['masks'] = F.vflip(dict['masks'])


def load_image(image_path, image_color_mode, transforms):
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

            
       
def subset_partition(subset, seed, fraction, image_list, test_list, model_folder):
    '''
    Here we shuffle the data
    '''
    if seed:
        np.random.seed(seed)
        indices = np.arange(len(image_list))
        np.random.shuffle(indices)
        image_list = np.array(image_list)[indices.astype(int)]

    if len(test_list) == 0:
        '''
        Select the last fraction % of the list as train
        '''
        if subset == "Train":
            partition = image_list[:int(
                np.ceil(len(image_list) * (1 - fraction)))]
        else:
            partition = image_list[
                int(np.ceil(len(image_list) * (1 - fraction))):]
        #save the list of observations used in this phase
        with open(os.path.join(model_folder, f'{subset}_obs.pkl'), "wb") as f:
            pickle.dump(partition, f)
        
    else:
        if subset == "Train":
            set_all = set(image_list)
            set_test = set(test_list)
            partition = list(set_all - set_test)
        else:
            partition = test_list
            
    return partition



class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 masks_folders: list, # list of paths
                 image_list: list = [], # list of images to consider
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = 0.2,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale",
                 test_list: list = []) -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            masks_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.
        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)

        '''
        Define the folders correctly
        '''
        self.masks_folders = masks_folders
        image_folder_path = Path(self.root) / image_folder #La barra es un operador de la clase Path
        self.masks_folder_paths = [Path(self.root) / p for p in masks_folders]

        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        for p in self.masks_folder_paths:
            if not p.exists():
                raise OSError(f"{p} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        # If a list of images is given
        if len(image_list) == 0:
            image_list = sorted(image_folder_path.glob("*"))
        else:
            image_list = [Path(self.root) / 'Images' / p for p in image_list]

        if subset not in ["Train", "Test"]:
            raise (ValueError(
                f"{subset} is not a valid input. Acceptable values are Train and Test."
            ))
        '''
        Here we shuffle the data
        '''
        if seed:
            np.random.seed(seed)
            indices = np.arange(len(image_list))
            np.random.shuffle(indices)
            image_list = np.array(image_list)[indices.astype(int)]
        '''
        Select the last fraction % ot the list as train
        '''
        if len(test_list) == 0:
            if subset == "Train":
                self.image_names = image_list[:int(
                    np.ceil(len(image_list) * (1 - fraction)))]
            else:
                self.image_names = image_list[
                    int(np.ceil(len(image_list) * (1 - fraction))):]
        else:
            if subset == "Train":
                set_all = set(image_list)
                set_test = set(test_list)
                self.image_names = list(set_all - set_test)
            else:
                self.image_names = test_list


    def image_names(self) -> list:
        return self.image_names

    def __len__(self) -> int:
        return len(self.image_names)
    '''
    Given an index returns a dictionary in the form:
        {
            "image": original image tensor
            "masks": multidimensional tensor containing all the masks
        }
    '''

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        sample = {'name': str(image_path).split('/')[-1]}

        # Let's read the image
        with open(image_path, "rb") as image_file:
            '''
            Read the image and save it in the dictionary
            '''
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")

        sample['image'] = self.transforms(image)
        sample['masks'] = []


        # Let's create a black image for the masks that don't correspond to the image
        THRESHOLD_VALUE = 255
        #Load image and convert to greyscale
        imgData = np.asarray(image.convert("L"))
        black_mask = (imgData > THRESHOLD_VALUE) * 1.0
        empty_mask = -1 * np.ones(black_mask.shape)

        # Iterate over all the masks, read them and save them in the dictionary
        for mask_path in self.masks_folder_paths:
            try:
                with open(mask_path / image_path.parts[-1],"rb") as mask_file:
                    mask = Image.open(mask_file)
                    if self.mask_color_mode == "rgb":
                        mask = mask.convert("RGB")
                    elif self.mask_color_mode == "grayscale":
                        mask = mask.convert("L")
                    sample['masks'].append(self.transforms(mask))
            except:
                if image_path.parts[-1][-7:-5] == mask_path.parts[-1][-3:]:
                    sample['masks'].append(self.transforms(empty_mask))
                else:
                    sample['masks'].append(self.transforms(black_mask))
        sample['masks'] = torch.cat(sample['masks'], 0).float() # Concatenate the masks in one unique tensor
        return sample

    
    
######################################################################################################


def input_boolean_net(images_folder_path, image_name, image_color_mode, transforms, flip=True,                                 dil=False):
    lat_image_path = str(images_folder_path / image_name) + '_lat.jpg'
    dor_image_path = str(images_folder_path / image_name) + '_dor.jpg'

    lat_image = load_image(lat_image_path, image_color_mode,transforms)
    dor_image = load_image(dor_image_path,image_color_mode,transforms)
    if flip:
        p = 0.5
        #Random flips over the lateral image
        if torch.rand(1) < p:
            lat_image = F.hflip(lat_image)
        if torch.rand(1) < p:
            lat_image = F.vflip(lat_image)
        #Random flips over the dorsal image
        if torch.rand(1) < p:
            dor_image = F.hflip(dor_image)
        if torch.rand(1) < p:
            dor_image = F.vflip(dor_image)
        #Cat dorsal-lateral or lateral-dorsal
        if torch.rand(1) < p:
            inp = 255*torch.cat((lat_image, dor_image), 1)
        else:
            inp = 255*torch.cat((dor_image,lat_image), 1)
    else:
        inp = 255*torch.cat((lat_image, dor_image), dim=1)

    return inp


class ClassificationDataset(VisionDataset):
    """A PyTorch dataset for image classification task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to the Images and transforms_masks to the masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 feno_names: list,
                 model_folder: str,
                 image_list: list = [],
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 test_list: list = []) -> None:
        
        self.subset = subset
        self.transforms = transforms
        self.root = root
        self.feno_names = feno_names
        self.image_folder_path = Path(self.root) / image_folder #La barra es un operador de la clase Path

        if not self.image_folder_path.exists():
            raise OSError(f"{self.image_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if subset not in ["Train", "Test"]:
            raise ValueError(
                f"{subset} is not a valid input. Acceptable values are Train and Test."
            )
        self.image_color_mode = image_color_mode
        self.image_names = subset_partition(subset, seed, fraction, image_list, test_list, model_folder)

        
    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_name = self.image_names[index].split('/')[-1]
        input = input_boolean_net(self.image_folder_path,
                                  image_name,
                                  self.image_color_mode,
                                  self.transforms,
                                  flip=(self.subset=="Train"))

        #Concatenate the obtained masks:
        sample = {'image' : input}

        #get the boolean fenotypes from the stats.csv:
        stats = pd.read_csv(self.root + '/stats.csv')
        stats_image = stats[stats['image_name'] == image_name]
        fenotypes = {}
        for feno in self.feno_names:
            fenotypes[feno] = list(stats_image[feno])[0]
        sample['fenotypes'] = fenotypes

        return sample

