import os
from PIL import Image, ImageTk
import os
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import numpy as np


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



def print_mask(filename):
    matplotlib.use('agg')
    device = 'cpu'
    model = torch.load('static/weight/weights.pt' , map_location = torch.device('cpu'))
    img = Image.open('/Users/hugo.jimenez/Desktop/20190902_1046_R1_1056_R1_tail_W_right_G02_1_2.png')
    data_transforms = transforms.Compose([transforms.ToTensor()])
    data_img = data_transforms(img)
    inputs = data_img.unsqueeze(0).to(device)
    outputs = model(inputs)
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    for i in range(6):
        mask = torch.split(outputs['out'].cpu(),1,0)[0].squeeze()[i].detach().numpy()
        mask = np.ma.masked_where(mask < 0.1, mask)
        plt.imshow(torch.split(data_img.unsqueeze(0),1,0)[0].squeeze().permute(1, 2, 0), alpha=0.7)
        plt.imshow(mask, cmaps[i], alpha = 0.7)
        part = name(i)
        plt.savefig(filename[:-4]+'_'+ part +'_out.png', bbox_inches='tight')
