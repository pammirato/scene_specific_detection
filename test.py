from TrainData import *
import matplotlib.pyplot as plt
import numpy as np

a = TrainData('Bedroom_01_1')

for il in range(0, 50):
  bat = a.next_batch(50)
  print il



imgs = bat[0]
labels = bat[1]




#for il in range(0, imgs.shape[0]):
#
#  img = imgs[il,:,:,:]
#  label = np.nonzero(labels[il,:])
#
#  plt.imshow(img.astype(np.uint8)) 
#  plt.title(str(label))
#  plt.show()
#
