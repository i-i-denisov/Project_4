#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
#import pickle
#import glob
import config



def load_dataset():
    filenames = []
    zero_center_images = []
    steering_angles = []
    throttle_positions = []
    brake_positions = []
    speed_values = []
    with open(config.filepath[config.environment] +config.filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header=reader.__next__()
        if header!=config.expected_header:
            raise Exception("Unexpected header in data file ", config.filename)
        #zero_steering_frames=0
        for row in reader:
            angle=float(row[3])
            if config.use_center_cam:
                filenames.append(config.filepath[config.environment] + row[0].strip())
                steering_angles.append(angle)
            if config.use_left_cam:
                filenames.append(config.filepath[config.environment] + row[1].strip())
                steering_angles.append(angle+config.left_camera_steering_offset)
            if config.use_right_cam:
                filenames.append(config.filepath[config.environment] + row[2].strip())
                steering_angles.append(angle + config.right_camera_steering_offset)
            #throttle_positions.append(float(row[4]))
            #brake_positions.append(float(row[5]))
            #speed_values.append(float(row[6]))
    return filenames,steering_angles

def gen_images_load(image_files,labels,training=True,batch_size = 32):
    while True:
        if config.mirror_augment_enable & training:
            batch_size=int(batch_size/config.batch_factor[config.mirror_augment_enable])
        dataset_size = len(labels)
        assert dataset_size==len(image_files), "mismatch between images and labels number"
        for batch in range(0,dataset_size,batch_size):
            batch_files,batch_labels=image_files[batch:batch+batch_size],labels[batch:batch+batch_size]
            batch_images=images_load(batch_files)
            #print(f'loaded {len(batch_files)} images')
            #print(f'image array size:{batch_images.shape}')
            if config.mirror_augment_enable & training:
                aug_images,aug_labels=augment_w_mirror(batch_images,batch_labels)
                batch_images=np.append(batch_images,aug_images,axis=0)
                batch_labels=np.append(batch_labels,aug_labels)
                #print (f' imageset augmented with mirroring')
                #print (f' new image array size is: {batch_images.shape}')
            yield batch_images,batch_labels

def augment_dataset(images , labels,visualise=False): # TODO: write a code that can be reused for adding types of augmentations to be used as an input
    dataset_size = len(images)
    aug_images=np.zeros([images.shape[0]*2,images.shape[1],images.shape[2],images.shape[3]],dtype=images.dtype)
    aug_labels=np.zeros(len(labels)*2,dtype=type(labels[0]))
    aug_images[:images.shape[0], :, :] = images
    aug_labels[:images.shape[0]] = labels
    aug_images[images.shape[0]:, :, :], aug_labels[images.shape[0]:]=augment_w_mirror(images,labels)
    print("Dataset augmented with mirror")
    if visualise:
        i=np.random.randint(dataset_size)
        visualise_mirroring(aug_images,i, dataset_size)
    return aug_images,aug_labels

def augment_w_mirror(images,labels):
    aug_images,aug_labels=np.flip(images,axis=2),np.negative(labels)
    return aug_images,aug_labels

def images_load(list):
    try:
        image = cv2.imread(list[0])
    except:
        print ("Error opening file ",list[0])
        return None
    images = np.empty((len(list),image.shape[0],image.shape[1],image.shape[2]),dtype='uint8')
    i=0
    for file in list:
        try:
            image=cv2.imread(file)
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print("Error opening file ", file)
            break
        image=np.expand_dims(image, axis=0)
        images[i, :, :,:] = image
        i+=1
    return images



def visualise_mirroring(images,i,dataset_size):
    fig, ax = plt.subplots(1, 2)
    #fig.suptitle(center_images[i], fontsize=8)
    ax[0].imshow(cv2.cvtColor(images[i + dataset_size], cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.show()