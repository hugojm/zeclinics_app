#!/usr/local/bin/python

from PIL import Image, ImageTk
import os
from torchvision import transforms
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np

import cgi, os
import cgitb; cgitb.enable()


form = cgi.FieldStorage()

fileitem = form['filename']

# check if the file has been uploaded
if fileitem.filename:
    # strip the leading path from the file name
    fn = os.path.basename(fileitem.filename)

   # open read and write the file into the server
    open(fn, 'wb').write(fileitem.file.read())

device = 'cpu'
model = torch.load('./sources/weights.pt' , map_location = torch.device('cpu'))
img = Image.open(fileitem.filename)
data_transforms = transforms.Compose([transforms.ToTensor()])
data_img = data_transforms(img)
inputs = data_img.unsqueeze(0).to(device)
outputs = model(inputs)
plt.imshow(torch.split(data_img.unsqueeze(0),1,0)[0].squeeze().permute(1, 2, 0))
cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
figure = Figure(figsize=(4,3), dpi=100)
subplot = figure.add_subplot(111)
subplot.imshow(torch.split(data_img.unsqueeze(0),1,0)[0].squeeze().permute(1, 2, 0), alpha=0.7)

for i in range(6):
    mask = torch.split(outputs['out'].cpu(),1,0)[0].squeeze()[i].detach().numpy()
    mask = np.ma.masked_where(mask < 0.1, mask)
    subplot.imshow(mask, cmaps[i], alpha = 0.7)
fig.savefig('sources/'+filename[:-4]+'.png')
