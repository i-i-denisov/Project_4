import keras
from numpy import zeros
imgsize=[240,320,3]
environment="PC"
force_dataset_reload=True
visualise_loading_dataset=True
save_images_as_array={"PC":True,"AWS":False}
mirror_augment_enable=True
freeze_weights=True
use_left_cam=True
use_right_cam=True
use_center_cam=True
left_camera_steering_offset=0.2
right_camera_steering_offset=-0.2
validation_split=0.2
batch_size = {'PC':32,'AWS':250}
batch_factor={True:2,False:1}
epochs = 3
dropout_rate=0.5
learning_rate=0.001
lr_decay=0.75
optimizer = keras.optimizers.Adam(lr=learning_rate)
loss=keras.losses.mse
metrics=[keras.metrics.mse]
filepath={'PC':'c:/Tools/Udacity/Project_4/data/','AWS':"/opt/carnd_p3/data/"}
filename="driving_log.csv"
images_pickle=filepath[environment]+"images.dump"
labels_pickle=filepath[environment]+"labels.dump"
expected_header=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
crop_mask=[70, 136, 60, 260]
input_resize_shape=[224, 224, 3] #pre-trained models can't work with custom image sizes this is why we have to bring our images to certain shape. shape defined here will be used througout training process.
model_savename='model.h5'
global batch_uses
batch_uses=zeros(1206,dtype=int)
