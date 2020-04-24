import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
from keras_facenet import FaceNet

def createDatabase():
	database = pd.DataFrame({'folder_id':[],'photo_id':[],'face_id':[],'embedding':[]},dtype="float32")

	#traverse through all the images
	images_path = "/home/darealappu/Desktop/CDAC/DR-GAN tensorflow/vggface2_test/test"
	index=0
	time_to_return = False
	embedder = FaceNet()

	for folder in os.listdir(images_path):
		folder_id = int(folder[1:])
		for image in os.listdir(os.path.join(images_path,folder)):
			if time_to_return == True:
				return database
			photo_id = int(image[0:4])
			face_id = int(image[5:7])
			img = cv2.imread(os.path.join(images_path,folder,image)).astype(np.float32)
			
			img.resize((1,img.shape[0],img.shape[1],img.shape[2]))
			
			em = embedder.embeddings(img)

			database=database.append({
				'folder_id':folder_id,
				'photo_id':photo_id,
				'face_id':face_id,
				'embedding':em},ignore_index=True)

			index+=1
			print(index-1)

			if index%10==0:
				if(index==5000):
					time_to_return = True
				break;
	return database



if __name__ == '__main__':
	database = createDatabase()
	database.to_pickle('database.csv')
