from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import numpy as np
import random


def convertRGB(image_dir):
    filenames = os.listdir(image_dir)
    num = np.array(filenames).shape[0]
    print(num)
    for i in range(num):
        image = Image.open(os.path.join(image_dir, filenames[i]))
        image_np = np.array(image)
        img_ch = np.array(image_np.shape)
        if img_ch.shape[0]==2:
            image.convert('RGB').save('./data/edges2shoes/convert/'+filenames[i])
            #image_rgb_np = np.array(image_rgb)
            #img_rgb_ch = np.array(image_rgb_np.shape)
            #print(img_rgb_ch.shape)

        #print(image)


image_dir = './data/edges2shoes/shoes'
convertRGB(image_dir)