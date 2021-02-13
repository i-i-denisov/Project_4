
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import functions
import config

import keras
from sklearn.model_selection import train_test_split

#import matplotlib.image as mpimg
import cv2





imagefiles,steering_angles=functions.load_dataset()
dataset_size=len(steering_angles)
#print ("Dataset length is ",dataset_size)
try:
    image = cv2.imread(imagefiles[0])
    imgshape = image.shape
    print(f'image shape is {imgshape}')
except:
    print("Error opening file ", imagefiles[0])
train_files, validate_files, train_y, validate_y=train_test_split(imagefiles,steering_angles,test_size=config.validation_split)
train_len=len(train_y)
validate_len=len(validate_y)
print (f' training images {len(train_files)} training labels {train_len}')
print (f' validation images {len(validate_files)} training labels {validate_len}')

train_gen=functions.gen_images_load(train_files,train_y)
validate_gen=functions.gen_images_load(validate_files,validate_y)


##trainig pipeline
inputs=keras.Input(imgshape)
#TODO crop and resize here
crop=keras.layers.Cropping2D(cropping=((config.crop_mask[0],imgshape[0]-config.crop_mask[1]),(config.crop_mask[2],imgshape[1]-config.crop_mask[3])))(inputs)
#print(crop)
normalize=keras.layers.Lambda(lambda x: (x-128)/128 )(crop)
#print(normalize)
conv1=keras.layers.Conv2D(24,5,strides=(2, 2),activation='relu')(normalize)
dropout_1=keras.layers.Dropout(rate=config.dropout_rate)(conv1)
#print (conv1)
conv2=keras.layers.Conv2D(36,5,strides=(2, 2),activation='relu')(dropout_1)
dropout_2=keras.layers.Dropout(rate=config.dropout_rate)(conv2)
#print (conv2)
conv3=keras.layers.Conv2D(48,5,strides=(2, 2),activation='relu')(dropout_2)
#print (conv3)
conv4=keras.layers.Conv2D(64,3,activation='relu')(conv3)
#print (conv4)
conv5=keras.layers.Conv2D(64,3,activation='relu')(conv4)
#print (conv5)
flat=keras.layers.Flatten()(conv5)
#print(flat)
FC1=keras.layers.Dense(100,activation='relu')(flat)
#FC_dropout_1=keras.layers.Dropout(rate=config.dropout_rate)(FC1)
FC2=keras.layers.Dense(50,activation='relu')(FC1)
#FC_dropout_2=keras.layers.Dropout(rate=config.dropout_rate)(FC2)
FC3=keras.layers.Dense(10,activation='relu')(FC2)
#FC_dropout_3=keras.layers.Dropout(rate=config.dropout_rate)(FC3)
steer=keras.layers.Dense(1,activation='linear')(FC3)
model=keras.Model(inputs=inputs, outputs=steer)
#model.summary()

batches_per_epoch = dataset_size/config.batch_size[config.environment]*config.batch_factor[config.mirror_augment_enable]
steps_per_epoch=dataset_size*config.batch_factor[config.mirror_augment_enable]
print(f'Dataset length is {dataset_size}, batch size is {config.batch_size[config.environment]}, mirror augmentation {config.mirror_augment_enable}\n batches per epoch {batches_per_epoch}' )
eqiuv_decay = (1./config.lr_decay -1)/batches_per_epoch

model.compile(loss=config.loss, optimizer=config.optimizer, metrics=config.metrics)
print(model.optimizer.get_config())
model.fit_generator(train_gen,
    epochs=config.epochs,
    steps_per_epoch=int(train_len*config.batch_factor[config.mirror_augment_enable]/config.batch_size[config.environment]),
    verbose=1,
    callbacks=None,
    validation_data=validate_gen,
    validation_steps=int(validate_len*config.batch_factor[config.mirror_augment_enable]/config.batch_size[config.environment])
)
model.save(config.model_savename)