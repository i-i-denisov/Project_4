import keras
force_dataset_reload=True
visualise_loading_dataset=False
mirror_augment_enable=False
freeze_weights=True
zero_steering_angle_frames_limit=500
validation_split=0.2
batch_size = 192
epochs = 30
dropout_rate=0.5
learning_rate=0.01
optimizer = keras.optimizers.Adam(lr=learning_rate)
loss=keras.losses.mse
metrics=[keras.metrics.mae]
filepath="/home/workspace/CarND-Behavioral-Cloning-P3/drive_data/data/"
filename="driving_log.csv"
images_pickle=filepath+"images.dump"
labels_pickle=filepath+"labels.dump"
expected_header=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
crop_mask=[70, 136, 60, 260]
input_resize_shape=[224, 224, 3] #pre-trained models can't work with custom image sizes this is why we have to bring our images to certain shape. shape defined here will be used througout training process.
model_savename='model.h5'