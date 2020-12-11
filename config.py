import tensorflow.keras as keras
force_dataset_reload=False
visualise_loading_dataset=False
freeze_weights=True
zero_steering_angle_frames_limit=500
validation_split=0.2
batch_size = 32
epochs = 20
dropout_rate=0.5
learning_rate=0.01
optimizer = keras.optimizers.Adam(lr=learning_rate)
loss=keras.losses.mse
metrics=[keras.metrics.mae]
filepath="C:/Tools/Udacity/Project_4/data/"
filename="driving_log.csv"
images_pickle=filepath+"images.dump"
labels_pickle=filepath+"labels.dump"
expected_header=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
image_below_horizon_mask=[70,160,0,320]
input_resize_shape=[224, 224, 3] #pre-trained models can't work with custom image sizes this is why we have to bring our images to certain shape. shape defined here will be used througout training process.
model_savename='model.h5'