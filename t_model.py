import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
# import network
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch 
import numpy as np
from tqdm import tqdm
import kornia
import dataset
from torch.nn import functional as F
from torchvision.utils import save_image
import sir9
import datetime
import cv2
import os

# 指定使用0,1,2三块卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cpu")


my_model = sir9.HDRPointwiseNN()
# my_model.cuda()

# torch.load(map_loction=torch.device('cpu'))
my_model.load_state_dict(torch.load("model/our_deblursir9_520.pth",map_location=lambda storage, loc: storage))
# my_model.eval()
#GAN.load_state_dict(torch.load("/home/dell/IJCAI/JBL/JBPSC/model/model_g_epoch69.pth"))
to_pil_image = transforms.ToPILImage()


tfs_full = transforms.Compose([
            #transforms.Resize(1080),
            transforms.ToTensor()
        ])



def load_simple_list(src_path):
    name_list = list()
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = [name for name in name_list if '.jpg' in name]
    name_list.sort()
    return name_list
   
#list_s = load_simple_list('/6T/home/dell/PyTorch-Image-Dehazing-master/O-HAZY/hazy/35_outdoor_hazy.jpg') 

'''
i = 0
'''
#for idx in range(len(list_s)):

image_in = Image.open('test/24349.JPG').convert('RGB')
#content = cv2.imread('/6T/home/dell/PyTorch-Image-Dehazing-master/O-HAZY/hazy/35_outdoor_hazy.jpg') 

#content = content.transpose((2, 0, 1))/255.0
#content = torch.tensor(content).unsqueeze(0).float().cuda()

full = tfs_full(image_in).unsqueeze(0).to(device)
  
  #hg, wg = torch.meshgrid([torch.arange(0, full.shape[2]), torch.arange(0, full.shape[3])]) # [0,511] HxW

  #hg = hg.to(device)
  #wg = wg.to(device)

start = time.time()
output = my_model(full)
end = time.time()
print(end - start)


  #save_image(output[0], '{}'.format(list_s[idx]))
save_image(output, 't_result/24349sys.JPG'.format(0+1))



