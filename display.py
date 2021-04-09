import os
from PIL import Image, ImageTk
import os
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import numpy as np

def print_mask(filename):
    matplotlib.use('agg')
    device = 'cpu'
    model = torch.load('static/weight/weights.pt' , map_location = torch.device('cpu'))
    img = Image.open(filename)
    data_transforms = transforms.Compose([transforms.ToTensor()])
    data_img = data_transforms(img)
    inputs = data_img.unsqueeze(0).to(device)
    outputs = model(inputs)
    plt.imshow(torch.split(data_img.unsqueeze(0),1,0)[0].squeeze().permute(1, 2, 0))
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    plt.imshow(torch.split(data_img.unsqueeze(0),1,0)[0].squeeze().permute(1, 2, 0), alpha=0.7)
    for i in range(6):
        mask = torch.split(outputs['out'].cpu(),1,0)[0].squeeze()[i].detach().numpy()
        mask = np.ma.masked_where(mask < 0.1, mask)
        plt.imshow(mask, cmaps[i], alpha = 0.7)
    plt.savefig(filename[:-4]+'_out.png', bbox_inches='tight')
