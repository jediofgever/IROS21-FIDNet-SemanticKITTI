import os
import numpy as np
from PIL import Image, ImageOps
import glob
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import POSSDataset

#A = POSSDataset.POSSDataset(root="/home/atas/poss_data/", split='train')
A = POSSDataset.POSSDataset(root="/home/atas/IROS21-FIDNet-SemanticKITTI/poss_data/", split='test')

# print(A[0][0].shape) # x,y,z,intensity,radius,nx,ny,nz 8 , 128, 2048
# print(A[0][1].shape) # semantic label 1, 128, 2048
# print(A[0][2].shape) # semantic label mask 1, 128, 2048

it = A[0][0][5:8]
it = np.transpose(it)

print(it.shape)
it = np.swapaxes(it,0,1)
print(it.shape)
 
plt.imshow(it, interpolation='nearest')
plt.show()

'''
data_loader_train = torch.utils.data.DataLoader(A,batch_size=4,shuffle=True,num_workers=4,pin_memory=True,drop_last=True,collate_fn=lambda x: x)

for batch_ndx, sample in enumerate(data_loader_train):
	print (batch_ndx)
	print (sample.keys())
'''
