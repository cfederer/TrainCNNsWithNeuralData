# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np

#from tensorflow.contrib.data import Dataset
from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

#IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGeneratorV(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, img_size=227, 
                 buffer_size=1000):

        self.txt_file = txt_file

        # retrieve the data from the text file
        self._read_txt_file()

        self.img_size = img_size

        # number of samples in the dataset
        self.data_size = len(self.RSAs)
        self.batch_size = batch_size 


        # convert lists to TF tensor
        self.img1_paths = convert_to_tensor(self.img1_paths, dtype=dtypes.string)
        self.img2_paths = convert_to_tensor(self.img2_paths, dtype=dtypes.string)
        self.RSAs = convert_to_tensor(self.RSAs, dtype=dtypes.float32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img1_paths,self.img2_paths, self.RSAs))

        data = data.map(self._parse_function_train)

        data = data.batch(batch_size)

        self.data = data 
        
    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img1_paths = []
        self.img2_paths = []
        self.RSAs = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                self.img1_paths.append(items[0])
                self.img2_paths.append(items[1])
                self.RSAs.append(float(items[2]))

    def _parse_function_train(self, filename1, filename2, RSA):
        """Input parser for samples of the training set."""
        # load and preprocess the image
        img1_string = tf.read_file(filename1)
        img1_decoded = tf.image.decode_png(img1_string, channels=3)
        img1_resized = tf.image.resize_images(img1_decoded, [int(self.img_size), int(self.img_size)])

        img2_string = tf.read_file(filename2)
        img2_decoded = tf.image.decode_png(img2_string, channels=3)
        img2_resized = tf.image.resize_images(img2_decoded, [int(self.img_size), int(self.img_size)])
        """
        Dataaugmentation comes here.
        """
        # RGB -> BGR
        img_bgr1 = img1_resized[:, :, ::-1]
        img_bgr2 = img2_resized[:, :, ::-1]
        
        return img_bgr1, img_bgr2, RSA
         
            
    def _file_to_img(self, filename):
        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [32, 32])
        #img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_resized[:, :, ::-1]

        return img_bgr
