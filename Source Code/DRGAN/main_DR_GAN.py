import os
import scipy.misc
import numpy as np
import cv2

from DR_GAN_model import DR_GAN
import tensorflow as tf

#new ones
#import os
import pandas as pd

flags = tf.app.flags
flags.DEFINE_boolean("return_img", True, "True for return img, False for feature and coefficient only, img is set ot 0")
flags.DEFINE_string("gpu", "0", "GPU to use [0]")
flags.DEFINE_string("input", "./img_in.JPG", "Path to input")
flags.DEFINE_string("output_prefix", "./out_", "Path to output")
FLAGS = flags.FLAGS


def main(_):
    gpu_options = tf.GPUOptions(visible_device_list =FLAGS.gpu, per_process_gpu_memory_fraction=0.5)
    dr_gan = DR_GAN(gpu_options = gpu_options)
    
    path = FLAGS.input
    if os.path.isfile(path):
        # Read image
        img = cv2.imread(path).astype(np.float32)
        img = cv2.resize(img,(96,96))

        # Feed to DR-GAN
        rotated_image, feature, coefficient  = dr_gan.test(img, return_img = FLAGS.return_img)
        
        rotated_image = cv2.cvtColor(rotated_image,cv2.COLOR_BGR2RGB)
        cv2.imshow('img', rotated_image)
        cv2.waitKey(0)

        print(rotated_image)

        # Save output
        cv2.imwrite(FLAGS.output_prefix + 'img.png', rotated_image) 
        print('Frontalized image: ' +FLAGS.output_prefix + 'img.png' )
        np.savetxt(FLAGS.output_prefix + 'feature.txt', feature)
        print('Feature: ' +FLAGS.output_prefix + 'feature.txt' )
        np.savetxt(FLAGS.output_prefix + 'coefficient.txt', coefficient)
        print('Coefficient: ' +FLAGS.output_prefix + 'coefficient.txt' )
    elif os.path.isdir(path):
        # TO DO: remove pass
        pass
        # Read all images in the folder
        
        # Feed each image to the network

        # Save output as you need
        


if __name__ == '__main__':
    tf.app.run()
