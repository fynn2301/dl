import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        with open(label_path) as json_file:
            labels = {int(k): v for k, v in json.load(json_file).items()}
        images = {k: np.load(file_path+ '/' + str(k) + '.npy') for k in labels.keys()}
        
        self.images = np.asarray([resize(images[i], (image_size[0],image_size[1])) for i in range(len(images))])
        self.labels = np.asarray([labels[i] for i in range(len(labels))])
        
        self._shuffle_rotation_mirror()
        
        self.epoch = 0
        self.next_image = 0
        self.started = True
        
        
        

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        
        # check if the end is reached
        if self.next_image < self.batch_size and not self.started:
            self.epoch += 1
        
        self.started = False
        
        if self.next_image + self.batch_size >= len(self.images):
            needed_head = self.batch_size - (len(self.images) - self.next_image)
            images = self.images[self.next_image:]
            labels = self.labels[self.next_image:]
            
            self._shuffle_rotation_mirror()
            
            images_top =  self.images[: needed_head]
            labels_top =  self.labels[: needed_head]
            
            if images_top.size != 0:
                labels = np.concatenate((labels, labels_top))
                images = np.concatenate((images, images_top))
                
            self.next_image = needed_head
        else:
            images = self.images[self.next_image:self.next_image+self.batch_size]
            labels = self.labels[self.next_image:self.next_image+self.batch_size]
            self.next_image += self.batch_size
        return images, labels

    def _shuffle_rotation_mirror(self):
        # shuffel the list
        if self.mirroring:
            bool_array = np.random.randint(0, 2, len(self.images))
            indices_to_flip = np.where(bool_array == 1)
            self.images[indices_to_flip,:,:] = self.images[indices_to_flip,:,::-1]
        if self.rotation:
            rot_array = np.random.randint(0, 4, len(self.images))
            indices_rot_90 = np.where(rot_array == 1)
            indices_rot_180 = np.where(rot_array == 2)
            indices_rot_270 = np.where(rot_array == 3)
            
            # flip the 90 deg indices once
            self.images[indices_rot_90,:,:] = np.rot90(self.images[indices_rot_90,:,:], k=1, axes=(2, 3))# self.images[indices_to_flip,:,::-1]
            
            # flip the 180 deg indices twice
            self.images[indices_rot_180,:,:] = np.rot90(self.images[indices_rot_180,:,:], k=2, axes=(2, 3))# self.images[indices_to_flip,:,::-1]
            
            # flip the 270 deg indices three times
            self.images[indices_rot_270,:,:] = np.rot90(self.images[indices_rot_270,:,:], k=3, axes=(2, 3))# self.images[indices_to_flip,:,::-1]
            
        if self.shuffle:
            p = np.random.permutation(len(self.images))
            self.images = self.images[p]
            self.labels = self.labels[p]

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        

        return img
    
    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images, labels = self.next()
        fig = plt.figure()
        
        w = 5
        h = int(self.batch_size // w)
        i = 1
        for image, label in zip(images, labels):
            fig.add_subplot(h, w, i)
            plt.imshow(image)
            plt.axis('off')
            plt.title(self.class_dict[label])
            i += 1
        plt.show()
        pass
