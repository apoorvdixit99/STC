import time
import numpy as np
import pandas as pd

db = pd.read_pickle('database.csv') #FaceNet Features
feature = db['embedding']

NUMBER_OF_SAMPLES = 5000
NUMBER_OF_FEATURES = 512 #320 for DR-GAN

fnp = np.ndarray(shape=(NUMBER_OF_SAMPLES,NUMBER_OF_FEATURES))
for f in range(NUMBER_OF_SAMPLES):
  for e in range(NUMBER_OF_FEATURES):
    fnp[f][e] = feature[f][0][e]

from sklearn.manifold import TSNE
tsne = TSNE().fit_transform(fnp)

import matplotlib.pyplot as plt
plt.scatter(tsne[:,0],tsne[:,1],s=1)
plt.show()
