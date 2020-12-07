
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import functions
import config
import os.path
import tensorflow.keras
import csv
import matplotlib.image as mpimg

center_images=[]
steering_angles=[]
throttle_positions=[]
brake_positions=[]
speed_values=[]
if  (os.path.isfile(config.images_pickle) and os.path.isfile(config.images_pickle)) and (not config.force_dataset_reload):
    images_file=open(config.images_pickle,'rb')
    labels_file=open(config.labels_pickle,'rb')
    images=np.load(images_file)
    steering_angles=np.load(labels_file)
    print ("Loaded dataset from files",config.images_pickle," ,",config.labels_pickle)
    print("Dataset of ", images.shape[0], "x", images[0].shape, "dtype=", images.dtype, " images")
    images_file.close()
    labels_file.close()
else:
    with open(config.filepath+config.filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header=reader.__next__()
        if header!=config.expected_header:
            raise Exception("Unexpected header in data file ", config.filename)
        zero_steering_frames=0
        for row in reader:
            angle=float(row[3])
            if angle!=0:#discarding all zero angle images
                center_images.append(config.filepath+row[0])
                steering_angles.append(angle)
                throttle_positions.append(float(row[4]))
                brake_positions.append(float(row[5]))
                speed_values.append(float(row[6]))
            else:
                zero_steering_frames += 1
                if zero_steering_frames<config.zero_steering_angle_frames_limit:
                    center_images.append(config.filepath + row[0])
                    steering_angles.append(angle)
                    throttle_positions.append(float(row[4]))
                    brake_positions.append(float(row[5]))
                    speed_values.append(float(row[6]))



    images=functions.images_load(center_images)
    dataset_size=len(images)

    images=functions.preprocess_images(images,config.visualise_loading_dataset)
    images,steering_angles=functions.augment_dataset(images,steering_angles,config.visualise_loading_dataset)
    print ("Dataset of ",images.shape[0], "x", images[0].shape, "dtype=", images.dtype, " images")
    #saving dataset to pickle
    images_file=open(config.images_pickle,'wb')
    labels_file=open(config.labels_pickle,'wb')
    np.save(images_file,images)
    np.save(labels_file,steering_angles)
    print("Saved dataset to files", config.images_pickle, " ,", config.labels_pickle)
    images_file.close()
    labels_file.close()

if config.visualise_loading_dataset:
    hist, bins=np.histogram(steering_angles,100,[-1,1])
    plt.hist(steering_angles, bins)
    plt.show()

##trainig pipeline
inputs=tf.keras.Input(config.model_input_shape)
MobileNET=tf.keras.applications.MobileNetV2(
    input_shape=config.model_input_shape, alpha=0.35, include_top=False, weights='imagenet',
    input_tensor=inputs, pooling="avg")
if config.freeze_weights:
    for layer in MobileNET.layers:
            layer.trainable=False
FC1=tf.keras.layers.Dense(100,activation='relu')(MobileNET.output)
FC_dropout_1=tf.keras.layers.Dropout(rate=config.dropout_rate)(FC1)
FC2=tf.keras.layers.Dense(10,activation='relu')(FC_dropout_1)
FC_dropout_2=tf.keras.layers.Dropout(rate=config.dropout_rate)(FC2)
steer=tf.keras.layers.Dense(1,activation='relu')(FC_dropout_2)
model=tf.keras.Model(inputs=inputs, outputs=steer)
if config.visualise_loading_dataset:
    (model.summary())
    tf.keras.utils.plot_model(model, "cloning.png", show_shapes=True)
model.compile(loss=config.loss, optimizer=config.optimizer, metrics=config.metrics)
print(model.optimizer.get_config())
model.fit(x=images,
    y=steering_angles,
    batch_size=config.batch_size,
    epochs=config.epochs,
    verbose=1,
    callbacks=None,
    validation_split=config.validation_split,
    shuffle=True,
    use_multiprocessing=True,
)
model.save(config.model_savename,save_format='h5')