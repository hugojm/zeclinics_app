#
# VERSION 2.1
#

import time
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the model outside the function so it doesn't have to be loaded for every image
model = torch.load('./static/weight/weights_cardio.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict_heart_masks_deep(images,batch_size, model=model):
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
    masks = np.zeros(shape=(batch_size,482, 408), dtype=np.float32)
    for i in range(batch_size):
        mask = output_tensor['out'][i].cpu().detach().numpy()
        ret,masks[i] = cv2.threshold(mask,0.2,1,cv2.THRESH_BINARY)

    return masks

def nnet_masks(video, batch_size = 5,debug=False):
    # input shape is (batch_size, 482, 408, 3)
    if(debug):
        print("AI-Process starting...")
        start=time.time()

    masks = [None]*len(video)

    for i in range(0,len(video), batch_size):
        if (debug and (i+1)%100==0):
            print(" -> AI-Processing frame ",i+1," of ",len(video))

        masks[i:(i+batch_size)]=predict_heart_masks_deep(video[i:(i+batch_size)], batch_size)


    if debug:
        print("AI-PROCESS: Elapsed time = ",time.time()-start)

    return masks
# Example
#image = cv2.imread("/content/drive/MyDrive/Zeclinics/CARDIO/deep/Data/Image/20170104_SME02_008-0001.jpg")
#atrium_mask, ventricle_mask = predict_heart_masks_deep(image,model)
