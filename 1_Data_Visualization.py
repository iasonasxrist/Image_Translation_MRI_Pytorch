"""
Created on Fri Nov 12 15:46:50 2021

@author: iasonasxrist
"""

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import nibabel as nib
from glob import glob
import random
import os
from itkwidgets import view

path = "/Users/iasonasxrist/Documents/AIPYNB/small/"

# image = nib.load("/Users/iasonasxrist/Documents/AIPYNB/small/t1/IXI102-HH-1416-T1_fcm.nii.gz")
# image_data = image.get_fdata()
# print(image_data)


# Get the location of data

t1_dir = os.path.join(path,'t1')
t2_dir = os.path.join(path,'t2')

#select all docs with nii extension in specific folder

t1_fns = glob(os.path.join(t1_dir,'*.nii*'))
t2_fns = glob(os.path.join(t2_dir,'*.nii*'))

assert (len(t1_fns) ==len(t2_fns) and (t1_fns) !=0)


# Look at an axial view of the source T1-weighted (T1-w) and target T2-weighted (T2-w) images.
print(t1_fns[0])
print(t2_fns[0])


# Look at an axial view of the source T1-weighted (T1-w) and target T2-weighted (T2-w) images.

t1_ex,t2_ex = nib.load(t1_fns[0]).get_fdata() , nib.load(t2_fns[0]).get_fdata()
 
fig, (ax1,ax2) = plt.subplots(1,2 , figsize =(8,8))
ax1.imshow(np.rot90(t1_ex[:,:,50]),interpolation='nearest' , cmap='gray') ; 
ax1.set_title("T1-Weighted Image", fontsize=22); ax1.axis('off')

ax2.imshow(t2_ex[:,62,:],cmap = 'gray');
ax2.set_title("T2-Weighted Image",fontsize=22); ax2.axis('off')

view(t1_ex)
