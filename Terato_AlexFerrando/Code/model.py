from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch

################### MODELS FOR SEMANTIC SEGMENTATION ###################

def createDeepLabv3_resnet_50(outputchannels=6):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                                 progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    return model


def createDeepLabv3_mobilenet(outputchannels=6):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True,
                                     progress=True)
    model.classifier = DeepLabHead(960, outputchannels)
    return model


def createDeepLabv3_resnet_101(outputchannels=6):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                                 progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    return model




################### MODELS FOR BOOLEAN PHENOTYPES ###################

def binary_fenotypes_resnet101(outputchannels=10, pretrained=True):

    model = models.resnet101(pretrained=pretrained,
                                     progress=True)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, outputchannels)
    return model


def binary_fenotypes_resnet50(outputchannels=10, pretrained=True):

    model = models.resnet50(pretrained=pretrained,
                                     progress=True)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, outputchannels)
    return model


def binary_fenotypes_wideresnet50(outputchannels=10, pretrained=True):

    model = models.wide_resnet50_2(pretrained=pretrained,
                                     progress=True)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, outputchannels)
    return model


def binary_fenotypes_resnext101(outputchannels=10, pretrained=True):

    model = models.resnext101_32x8d(pretrained=pretrained,
                                     progress=True)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, outputchannels)
    return model


def binary_fenotypes_mobilenet_small(outputchannels=10, pretrained=True):

    model = models.mobilenet_v3_small(pretrained=pretrained,
                                     progress=True)
    '''
    Original header
      (classifier): Sequential(
    (0): Linear(in_features=576, out_features=1024, bias=True)
    (1):
    (2): Dropout(p=0.2, inplace=True)
    (3): Linear(in_features=1024, out_features=1000, bias=True)
  )
    '''
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=576, out_features=1024, bias=True),
        torch.nn.Hardswish(),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1024, out_features=outputchannels, bias=True)
    )

    return model

