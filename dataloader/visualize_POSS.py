import os
import numpy as np
from PIL import Image, ImageOps
import glob
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from laserscan import SemLaserScan, LaserScan
import yaml

from read_data import read_semlabels

root = '/home/atas/poss_data/'
split = 'test'

CFG = yaml.safe_load(open('/home/atas/IROS21-FIDNet-SemanticKITTI/poss_data/semantickitti19.yaml', 'r'))

color_dict = CFG["color_map"]
label_transfer_dict = CFG["learning_map"]
nclasses = 14

A = SemLaserScan(nclasses=nclasses, sem_color_dict=color_dict,
                 project=True, H=64, W=512, fov_up=25.0, fov_down=-30.0)
B = SemLaserScan(nclasses=nclasses, sem_color_dict=color_dict,
                 project=True, H=64, W=512, fov_up=25.0, fov_down=-30.0)

con_office_lidar = "/home/atas/IROS21-FIDNet-SemanticKITTI/poss_data/test/07/velodyne/000308.bin"
con_office_lidar_label = "/home/atas/IROS21-FIDNet-SemanticKITTI/poss_data/test/07/labels/000308.label"

labl = read_semlabels(con_office_lidar_label)

print(np.unique(labl))

A.open_scan(con_office_lidar)
A.open_label(con_office_lidar_label)

 
f, axarr = plt.subplots(2,1) 
axarr[0].imshow(A.proj_range)
axarr[1].imshow(A.proj_sem_color)

plt.show()
