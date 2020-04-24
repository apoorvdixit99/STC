import os
import scipy.misc
import numpy as np
import cv2
import pandas as pd
from DR_GAN_model import DR_GAN
import tensorflow as tf

def createDatabase():
	gpu_options = tf.GPUOptions(visible_device_list="0", per_process_gpu_memory_fraction=0.5)
	dr_gan = DR_GAN(gpu_options = gpu_options)
	database = pd.DataFrame({'folder_id':[],'photo_id':[],'face_id':[],'feature':[],'coefficient':[]},dtype="int64")

	#traverse through all the images
	images_path = "/home/darealappu/Desktop/CDAC/vggface2_test/test"
	index=0
	time_to_return = False

	for folder in os.listdir(images_path):
		folder_id = int(folder[1:])
		for image in os.listdir(os.path.join(images_path,folder)):
			if time_to_return == True:
				return database
			photo_id = int(image[0:4])
			face_id = int(image[5:7])
			img = cv2.imread(os.path.join(images_path,folder,image)).astype(np.float32)
			img = cv2.resize(img,(96,96))
			rotated_image, feature, coefficient  = dr_gan.test(img, return_img=True)

			database=database.append({
				'folder_id':folder_id,
				'photo_id':photo_id,
				'face_id':face_id,
				'feature':feature,
				'coefficient':coefficient},ignore_index=True)

			index+=1

			if index%10==0:
				if(index==5000):
					time_to_return = True
				break;
	return database





if __name__ == '__main__':
		print("Generation of Embeddings inititiated...")
		database = createDatabase()
		print("Embeddings successfully generated")
		database.to_pickle('database.csv')
