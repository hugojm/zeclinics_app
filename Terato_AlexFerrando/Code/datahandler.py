
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from segdataset import SegmentationDataset, ClassificationDataset

def get_dataloader_single_folder(data_dir: str,
                                 image_folder: str,
                                 masks_folders: list,
                                 fraction: float = 0.2,
                                 seed: int = None,
                                 batch_size: int = 4,
                                 num_workers: int = 8,
                                 images_list: list = [],
                                 test_list: list = []):
    
    """Create train and test dataloader from a single directory containing
    the image and mask folders.
    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        masks_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Test set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.
    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    
    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        x: SegmentationDataset(root=data_dir,
                               image_folder=image_folder,
                               masks_folders=masks_folders,
                               seed=seed,
                               fraction=fraction,
                               subset=x,
                               transforms=data_transforms,
                               image_list=images_list,
                               test_list=test_list)
        for x in ['Train', 'Test']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)
        for x in ['Train', 'Test']
    }
    return dataloaders, image_datasets


#######################################################################################################

    
def get_dataloader_single_folder_bool(data_dir: str,
                                 image_folder: str,
                                 feno_names: list,
                                 model_folder: str,
                                 image_list: list,
                                 seed: int,
                                 fraction: float = 0.2,
                                 batch_size: int = 4,
                                 num_workers: int = 8):
    """Create train and test dataloader from a single directory containing
    the image and mask folders.
    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        mask_folders (str, optional): Mask folder name. Defaults to 'Masks'.
    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        x: ClassificationDataset(data_dir,
                               image_folder = image_folder,
                               feno_names = feno_names,
                               model_folder = model_folder,
                               image_list = image_list,
                               seed = seed,
                               fraction = fraction,
                               subset = x,
                               transforms = data_transforms)
        for x in ['Train', 'Test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size = batch_size,
                      shuffle = True,
                      num_workers = num_workers)
        for x in ['Train', 'Test']
    }
    return dataloaders, image_datasets
