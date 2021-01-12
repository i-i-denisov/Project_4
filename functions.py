#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
#import pickle
#import glob
import config

def load_dataset_from_file():
    images_file = open(config.images_pickle, 'rb')
    labels_file = open(config.labels_pickle, 'rb')
    images = np.load(images_file)
    steering_angles = np.load(labels_file)
    print("Loaded dataset from files", config.images_pickle, " ,", config.labels_pickle)
    print("Dataset of ", images.shape[0], "x", images[0].shape, "dtype=", images.dtype, " images")
    images_file.close()
    labels_file.close()
    return images,steering_angles

def load_dataset_from_imageset():
    center_images = []
    zero_center_images = []
    steering_angles = []
    throttle_positions = []
    brake_positions = []
    speed_values = []
    with open(config.filepath+config.filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header=reader.__next__()
        if header!=config.expected_header:
            raise Exception("Unexpected header in data file ", config.filename)
        #zero_steering_frames=0
        for row in reader:
            angle=float(row[3])
            if (angle==0)and(config.discard_zero_steering_angles):
                zero_center_images.append(config.filepath+row[0])
            else:
                if config.use_center_cam:
                    center_images.append(config.filepath+row[0].strip())
                    steering_angles.append(angle)
                if config.use_left_cam:
                    center_images.append(config.filepath+row[1].strip())
                    steering_angles.append(angle+config.left_camera_steering_offset)
                if config.use_right_cam:
                    center_images.append(config.filepath+row[2].strip())
                    steering_angles.append(angle + config.right_camera_steering_offset)
            #throttle_positions.append(float(row[4]))
            #brake_positions.append(float(row[5]))
            #speed_values.append(float(row[6]))
    if not config.discard_zero_steering_angles:
        center_images.extend (zero_center_images)
        zero_angles=np.zeros(len(zero_center_images),dtype=np.float)
        steering_angles.extend(zero_angles)
    images=images_load(center_images)
    return images,steering_angles

def preprocess_images(images,visualise=True):
    images = mask_images(images, config.crop_mask)
    print ("Dataset images cropped to shape of ",images[0].shape," dtype=",images.dtype,)
    if visualise:
        plt.imshow(images[0])
        plt.show()
    images = dataset_resize(images, (config.input_resize_shape[0], config.input_resize_shape[1]))
    print("Dataset images resized to ",images[0].shape, " dtype=",images.dtype,)
    if visualise:
        plt.imshow(images[0])
        plt.show()
    images=dataset_integer_normalize(images)
    print("Dataset images normalized to range -128..128")
    if visualise:
        plt.imshow(images[0])
        plt.show()
    return images

def dataset_integer_normalize(x):
    #very rough image normalisation
    array=np.zeros_like(x,dtype=np.int8)
    array[:]=x[:]-128 #TODO: use CUDA for array operations
    return array

def dataset_resize(images,dshape=[128,128]):
    resized_images=np.zeros((images.shape[0],dshape[0],dshape[1],images.shape[3]),dtype=images.dtype)
    i=0
    for image in images:
            resized_images[i]=cv2.resize(image,dshape)
            i+=1
    return resized_images

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
        except:
            print("Error opening file ", file)
            break
        image=np.expand_dims(image, axis=0)
        images[i, :, :,:] = image
        i+=1
    dataset_size = len(images)
    print("Loaded dataset of ", dataset_size, images[0].shape, "dtype=", images.dtype, " images")
    return images

def mask_images(images, mask=config.crop_mask):
    crop_imgs = images[:,mask[0]:mask[1], mask[2]:mask[3]]
    return crop_imgs

def visualise_mirroring(images,i,dataset_size):
    fig, ax = plt.subplots(1, 2)
    #fig.suptitle(center_images[i], fontsize=8)
    ax[0].imshow(images[i + dataset_size])
    ax[1].imshow(images[i])
    plt.show()