#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:36:21 2018

@author: lenovo
"""


from unet import *
from data import *



data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

# change folder name to training set folder 
#myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
myGene = trainGenerator(2,'/home/lenovo/Documents/Major Project/Project material/Dataset/Retinal/AV - DRIVE/AV_DRIVE_groundtruth/training/','images','matlab_unet',data_gen_args,save_to_dir = None)
#print(*myGene, sep = '\n')   ##batch size changed from 32 to 2
print(myGene)

model = unet()

## Can't find this function
##model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
model.fit_generator(myGene,steps_per_epoch=30,epochs=1)  #steps per epoch changed from 300 to 30

# change folder name to test set folder
#testGene = testGenerator("data/membrane/test")
#testGene = testGenerator("AV_DRIVE_groundtruth/test/images")
test_datagen = ImageDataGenerator(rescale=1./255)

testGene = test_datagen.flow_from_directory(
        'AV_DRIVE_groundtruth/test/images',
        target_size=(256, 256),
        batch_size=32)

#image_generator = image_datagen.flow_from_directory(
#        train_path,
#        classes = [image_folder],
#        class_mode = None,       
#        target_size = target_size,
#        batch_size = batch_size,
#        save_to_dir = save_to_dir,
#        save_prefix  = image_save_prefix,
#        seed = seed)

results = model.predict_generator(testGene,30,verbose=1)

print(results)
# change folder name
saveResult("AV_DRIVE_groundtruth/test/test2",results)


#import matplotlib.image as mpimg
#import scipy.ndimage
#import numpy as np 
#
#img = mpimg.imread('AV_DRIVE_groundtruth/training/images/21_training.tif')
#mask = mpimg.imread('AV_DRIVE_groundtruth/training/matlab_unet/21_training.png')
#
#num_class =3
#img = img / 255
#mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
#new_mask = np.zeros(mask.shape + (num_class,))
#flag_multi_class=1
#for i in range(num_class):
#            #for one pixel in the image, find the class in mask and convert it into one-hot vector
#            #index = np.where(mask == i)
#            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
#            #new_mask[index_mask] = 1
#            
#        new_mask[mask == i,i] = 1
#new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
#mask = new_mask
#        #x,y,z = scipy.ndimage.imread(filepath).shape
##        print x
##        print y
##        print z
#print mask.shape
#        
