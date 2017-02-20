import os
import numpy as np
from scipy import misc 
import scipy.io as sio
import random as rnd



BASE_PATH = '/playpen/ammirato/Data/RohitData/';
META_BASE_PATH = '/playpen/ammirato/Data/RohitMetaData/';
  


scene_name = 'Home_14_1'
scene_path = os.path.join(BASE_PATH, scene_name)
meta_path = os.path.join(META_BASE_PATH, scene_name)
cur_img_index = 0

#get image names (jpg or png)
image_path = os.path.join(meta_path, 'classification','images') 

file_names = os.listdir(image_path)
num_examples = len(file_names)



num_pixels = 0;
for image_name in file_names:

  img = misc.imread(os.path.join(image_path, image_name))

  num_pixels = num_pixels + img.shape[0]*img.shape[1]



num_pixels = float(num_pixels)

means =[0,0,0]
num_images = len(file_names)
for image_name in file_names:

  img = misc.imread(os.path.join(image_path, image_name))

  means = means + (img.sum(0).sum(0)/num_pixels)





vars =[0,0,0]
num_images = len(file_names)
for image_name in file_names:

  img = misc.imread(os.path.join(image_path, image_name))
  red = np.power((img[:,:,0] - means[0]),2)
  green = np.power((img[:,:,1] - means[1]),2)
  blue = np.power((img[:,:,2] - means[2]),2)

  img = np.stack((red,green,blue),2)

  vars = vars + (img.sum(0).sum(0)/num_pixels)


stds = np.sqrt(vars)





