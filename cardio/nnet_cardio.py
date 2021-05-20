#
# VERSION 2.0
#

import time
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the model outside the function so it doesn't have to be loaded for every image
# model = torch.load('./static/weight/weightsResNet50_2_masks.pt')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict_heart_masks_deep(images,batch_size, model=None):
    images=np.array(images)
    formatted_img=np.empty((batch_size,482,408,3))
    if len(images[0].shape)==2:
        for i in range(len(images)):
            to_8_bit=(255*(images[i]/np.max(images[i]))).astype('uint8')
            aux=np.dstack((np.zeros(images[i].shape,dtype='int'),to_8_bit,np.zeros(images[i ].shape,dtype='int')))
            formatted_img[i]=aux
        images=formatted_img

        del formatted_img

    if images.shape != (batch_size,482, 408, 3):
        #print("Image is not shape (482, 408, 3), reshaping")
        images = np.reshape(images,(batch_size, 482, 408, 3))
        #raise RuntimeError("Please, give images with shape (482, 408, 3)")

    # Transformation to get input size (batch, 3, 482, 408)
    im = torch.tensor(images.astype(np.float32)/255.0)
    im.view(batch_size,482,408,3)
    im = torch.swapaxes(torch.swapaxes(im, 2, 3), 1, 2)

    input_tensor = im.to(device)
    output_tensor = model(input_tensor)#.to(device)

    # Postprocess masks
    atrium_masks = np.zeros(shape=(batch_size,482, 408), dtype=np.float32)
    ventricle_masks = np.zeros(shape=(batch_size,482, 408), dtype=np.float32)
    for i in range(batch_size):
        atrium_mask = output_tensor['out'][i][0].cpu().detach().numpy()
        ventricle_mask = output_tensor['out'][i][1].cpu().detach().numpy()
        # Threshold the mask
        ret,atrium_masks[i] = cv2.threshold(atrium_mask,0.2,1,cv2.THRESH_BINARY)
        ret,ventricle_masks[i]  = cv2.threshold(ventricle_mask,0.2,1,cv2.THRESH_BINARY)

    return atrium_masks, ventricle_masks

def nnet_masks(video, batch_size = 20,debug=False):
    # input shape is (batch_size, 482, 408, 3)
    if(debug):
        print("AI-Process starting...")
        start=time.time()

    masks_a = [None]*len(video)
    masks_v = [None]*len(video)

    for i in range(0,len(video), batch_size):
        if (debug and (i+1)%100==0):
            print(" -> AI-Processing frame ",i+1," of ",len(video))

        masks_a[i:(i+batch_size)],masks_v[i:(i+batch_size)]=predict_heart_masks_deep(video[i:(i+batch_size)], batch_size)



    if debug:
        print("AI-PROCESS: Elapsed time = ",time.time()-start)

    return masks_a,masks_v
