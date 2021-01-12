
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import functions
import config
import os.path
import keras

import random
#import matplotlib.image as mpimg
#import cv2
#from keras.backend import tf as ktf



if  (os.path.isfile(config.images_pickle) and os.path.isfile(config.images_pickle)) and (not config.force_dataset_reload):
    images,steering_angles=functions.load_dataset_from_file()
    dataset_size = len(steering_angles)
else:
    images,steering_angles=functions.load_dataset_from_imageset()
    dataset_size=len(steering_angles)
    print ("Dataset length is ",dataset_size)
    #print (steering_angles)
    if config.mirror_augment_enable:
        images,steering_angles=functions.augment_dataset(images,steering_angles,config.visualise_loading_dataset)
    print ("Dataset of ",images.shape[0], "x", images[0].shape, "dtype=", images.dtype, " images")
    if config.save_images_as_array:
        #saving dataset to file
        images_file=open(config.images_pickle,'wb')
        labels_file=open(config.labels_pickle,'wb')
        np.save(images_file,images)
        np.save(labels_file,steering_angles)
        print("Saved dataset to files", config.images_pickle, " ,", config.labels_pickle)
        images_file.close()
        labels_file.close()

if config.visualise_loading_dataset:
    i=random.randint(0,dataset_size)
    plt.imshow(images[i])
    plt.show()
    hist, bins=np.histogram(steering_angles,100,[-1,1])
    plt.hist(steering_angles, bins)
    plt.show()

##trainig pipeline
imgshape=images[0].shape
inputs=keras.Input(imgshape)
#TODO crop and resize here
crop=keras.layers.Cropping2D(cropping=((config.crop_mask[0],imgshape[0]-config.crop_mask[1]),(config.crop_mask[2],imgshape[1]-config.crop_mask[3])))(inputs)
#print(crop)
normalize=keras.layers.Lambda(lambda x: (x-128)/128 )(crop)
#print(normalize)
conv1=keras.layers.Conv2D(24,5,strides=(2, 2),activation='elu')(normalize)
#print (conv1)
conv2=keras.layers.Conv2D(36,5,strides=(2, 2),activation='elu')(conv1)
#print (conv2)
conv3=keras.layers.Conv2D(48,5,strides=(2, 2),activation='elu')(conv2)
#print (conv3)
conv4=keras.layers.Conv2D(64,3,activation='elu')(conv3)
#print (conv4)
conv5=keras.layers.Conv2D(64,3,activation='elu')(conv4)
#print (conv5)
flat=keras.layers.Flatten()(conv5)
#print(flat)
FC1=keras.layers.Dense(100,activation='elu')(flat)
FC_dropout_1=keras.layers.Dropout(rate=config.dropout_rate)(FC1)
FC2=keras.layers.Dense(50,activation='elu')(FC_dropout_1)
FC_dropout_2=keras.layers.Dropout(rate=config.dropout_rate)(FC2)
FC3=keras.layers.Dense(10,activation='elu')(FC_dropout_2)
#FC_dropout_3=keras.layers.Dropout(rate=config.dropout_rate)(FC3)
steer=keras.layers.Dense(1,activation='linear')(FC3)
model=keras.Model(inputs=inputs, outputs=steer)
#model.summary()
if config.visualise_loading_dataset:
    (model.summary())
    tf.keras.utils.plot_model(model, "cloning.png", show_shapes=True)
batches_per_epoch = dataset_size/config.batch_size
eqiuv_decay = (1./config.lr_decay -1)/batches_per_epoch

model.compile(loss=config.loss, optimizer=config.optimizer, metrics=config.metrics)
print(model.optimizer.get_config())
model.fit(x=images,
    y=steering_angles,
    batch_size=config.batch_size,
    epochs=config.epochs,
    verbose=1,
    callbacks=None,
    validation_split=config.validation_split,
    shuffle=True
)
model.save(config.model_savename)