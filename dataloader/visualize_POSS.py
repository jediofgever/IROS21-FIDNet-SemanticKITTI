import os
import numpy as np
from PIL import Image, ImageOps
import glob
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from laserscan import SemLaserScan, LaserScan
import yaml

root = '/home/atas/poss_data/'
split = 'test'

CFG = yaml.safe_load(open(root+'poss.yaml', 'r'))

color_dict = CFG["color_map"]
label_transfer_dict = CFG["learning_map"]
nclasses = 14

A = SemLaserScan(nclasses=nclasses, sem_color_dict=color_dict,
                 project=True, H=64, W=512, fov_up=25.0, fov_down=-30.0)
B = SemLaserScan(nclasses=nclasses, sem_color_dict=color_dict,
                 project=True, H=64, W=512, fov_up=25.0, fov_down=-30.0)
C = SemLaserScan(nclasses=nclasses, sem_color_dict=color_dict,
                 project=True, H=64, W=512, fov_up=25.0, fov_down=-30.0)

lidarfile_path = "/home/atas/IROS21-FIDNet-SemanticKITTI/poss_data/test/07/velodyne/000300.bin"
lidarfile_path_poss = "/home/atas/poss_data/train/00/velodyne/000000.bin"
lidar_kitti = "/home/atas/17/velodyne/000000.bin"

A.open_scan(lidarfile_path)
B.open_scan(lidarfile_path_poss)
C.open_scan(lidar_kitti)

f, axarr = plt.subplots(3,1) 
axarr[0].imshow(A.proj_range)
axarr[1].imshow(B.proj_range)
axarr[2].imshow(C.proj_range)

plt.show()
