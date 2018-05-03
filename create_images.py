#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:14:59 2018

@author: alex
"""

from __future__ import division, print_function
import os
import numpy as np
import matplotlib.pyplot as plt

class create_images():
    
    def create_train_params():
        train_dir = '/home/alex/Λήψεις/pattern_rec/03_SpeakerSpoofing/data/DS_10283_3055/ASVspoof2017_V2_train_fbank'
            
        filenames = [i for i in os.listdir(train_dir) if i.endswith(".cmp")]
        train_params=[]
        train_params_value=[]
            
        for filename in filenames:
            #print(filename)
            cmp_filename = os.path.join(train_dir, filename)
                
            with open(cmp_filename, 'rb') as fid:
                cmp_data = np.fromfile(fid, dtype=np.float32, count=-1)
        
            cmp_data = cmp_data.reshape((-1, 64))  # where nfilt = 64
                
            n_frames=cmp_data.shape[0]
            #17X64 image creation
            for i in range(n_frames):
                if i<9:
                    ##
                    image=cmp_data[0:i,:]
                    for y in range(9-i):
                        image=np.concatenate((image, cmp_data[i:i+1,:]), axis=0) 
                    
                    image=np.concatenate((image, cmp_data[i+1:i+9,:]), axis=0)
                elif i>n_frames-9:
                        ##
                    image=cmp_data[i-9:i,:]
                    for y in range(i-(n_frames-8)):
                        image=np.concatenate((image,cmp_data[i:i+1,:]), axis=0)
                    
                    image=np.concatenate((image,cmp_data[i:n_frames,:]), axis=0)
                    
                else:
                    ##
                    image=cmp_data[i-8:i+9, :]
                    
                train_params=train_params + [image.T] #list of transposed images 64X17
                if filename<='T_1001508.cmp':
                    train_params_value=train_params_value + [1]
                else:
                    train_params_value=train_params_value + [0]

                #fbank = 10 # put a number between 0 and 63
                #plt.plot(image.T[fbank, :])
                #plt.show()
                
        return train_params, train_params_value
                   
    def create_dev_params():
        valid_dir = '/home/alex/Λήψεις/pattern_rec/03_SpeakerSpoofing/data/DS_10283_3055/ASVspoof2017_V2_train_dev'
        
        filenames = [i for i in os.listdir(valid_dir) if i.endswith(".cmp")]
        valid_params=[]  
        valid_params_value=[]
            
        for filename in filenames:
            #print(filename)
            cmp_filename = os.path.join(valid_dir, filename)
                
            with open(cmp_filename, 'rb') as fid:
                cmp_data = np.fromfile(fid, dtype=np.float32, count=-1)
        
            cmp_data = cmp_data.reshape((-1, 64))  # where nfilt = 64
                
            n_frames=cmp_data.shape[0]
            #17X64 image creation
            for i in range(n_frames):
                if i<9:
                    ##
                    image=cmp_data[0:i,:]
                    for y in range(9-i):
                        image=np.concatenate((image, cmp_data[i:i+1,:]), axis=0) 
                    
                    image=np.concatenate((image, cmp_data[i+1:i+9,:]), axis=0)
                elif i>n_frames-9:
                    ##
                    image=cmp_data[i-9:i,:]
                    for y in range(i-(n_frames-8)):
                        image=np.concatenate((image,cmp_data[i:i+1,:]), axis=0)
                    image=np.concatenate((image,cmp_data[i:n_frames,:]), axis=0)
                    
                else:
                    ##
                    image=cmp_data[i-8:i+9, :]
                    
                valid_params=valid_params + [image.T] #list of transposed images 64X17
                if filename<='D_1000760.cmp':
                    valid_params_value=valid_params_value + [1]
                else:
                    valid_params_value=valid_params_value + [0]
        
        return valid_params, valid_params_value