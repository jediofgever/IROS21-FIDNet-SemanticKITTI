import os
import numpy as np
from PIL import Image, ImageOps
import glob
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from laserscan import SemLaserScan,LaserScan
import yaml

root= '/home/atas/poss_data/'
split= 'test'

lidar_list = sorted(glob.glob(root+split+'/*/*/*.bin'))
label_list = [i.replace("velodyne", "labels") for i in lidar_list]
label_list = [i.replace("bin", "label") for i in label_list]
       
CFG = yaml.safe_load(open(root+'poss.yaml', 'r'))      
print(lidar_list[0])  
color_dict = CFG["color_map"]
label_transfer_dict =CFG["learning_map"]
nclasses = 14

A=SemLaserScan(nclasses=nclasses, sem_color_dict=color_dict, project=True, H=128, W=2048, fov_up=20.0, fov_down=-25.0)


for a in lidar_list:
    A.open_scan(a)
    plt.imshow(A.proj_range)
    plt.pause(0.05)
