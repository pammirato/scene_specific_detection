import os
import numpy as np
from scipy import misc 
import scipy.io as sio
import random as rnd


#TODO - allow more than one scene
#     -support multiple image sizes
#     -don't read all images at start

class InputData:
  'Organizes and provides training data'

  BASE_PATH = '/playpen/ammirato/Data/RohitData/';
  META_BASE_PATH = '/playpen/ammirato/Data/RohitMetaData/';
  

  input_max_dim = 224 
  num_cats = 2 

  
  IMG_CH = 3 


  def __init__(self,scene_name,jpg_imgs=1):
    "sets scene for this object, and initializes data array"
    self.scene_name = scene_name
    self.scene_path = os.path.join(InputData.BASE_PATH, scene_name)
    self.meta_path = os.path.join(InputData.META_BASE_PATH, scene_name)
    self.cur_img_index = 0

    self.init_data() 
  #init


  def get_number_of_images(self):
    return len(self.file_names)


  def init_data(self):
    "initializes data array with names of images to use"

    #get image names (jpg or png)
    self.image_path = os.path.join(self.meta_path, 'classification','images') 
    
    self.file_names = os.listdir(self.image_path)
    self.num_examples = len(self.file_names)

    #load all the images and labels
    #self.images = np.zeros((len(self.file_names), InputData.IMG_H, 
    #                        InputData.IMG_W,InputData.IMG_CH),dtype=np.int32)


    with open(os.path.join(self.meta_path,'classification', 'labels.txt')) as f:
      paths_and_labels = f.read().splitlines()          

    self.paths_and_labels = [str.split(s) for s in paths_and_labels]

    #randomly shuffle images   
    rnd.shuffle(self.paths_and_labels)

 

  def next_batch(self,batch_size,):
    "returns the next batch_size images"


    #get total number of training images
    num_imgs = len(self.paths_and_labels)

    #check to see if we need to wrap around to beginning  
    if (self.cur_img_index + batch_size) >= num_imgs :
      end_range = range(self.cur_img_index, num_imgs)
      start_range = range(0, batch_size - len(end_range)) 
      img_indices = end_range + start_range
    else:
      img_indices = range(self.cur_img_index, self.cur_img_index+batch_size)


    #make an array to hold all the images of the region proposals
    batch_imgs = np.zeros((len(img_indices), InputData.input_max_dim, 
                            InputData.input_max_dim,InputData.IMG_CH),dtype=np.float32)
   

    #hold ground truth labels for all images
    #gt_labels = np.zeros((cur_props.shape[0], InputData.num_cats)) 
    gt_labels = np.zeros((len(img_indices))) 

    #for each  image
    for il in range(0,len(img_indices)):
      index = img_indices[il]
      img_path = self.paths_and_labels[index][0];      
      label = int(self.paths_and_labels[index][1]);      
      full_img = misc.imread(os.path.join(img_path));    

      #normalize
      full_img[:,:,0] =  full_img[:,:,0] - 124
      full_img[:,:,1] =  full_img[:,:,1] - 117
      full_img[:,:,2] =  full_img[:,:,2] - 104
      full_img = full_img / 255.0

      #resize the full_img, but KEEP ASPECT RATIO
      scale_factor = InputData.input_max_dim / float(max(full_img.shape[0:2]));
      new_size = (int(full_img.shape[0]*scale_factor), int(full_img.shape[1]*scale_factor),3)
      resized_full_img = misc.imresize(full_img, new_size);

      #pad image with zeros to get to desired size
      blank_img = np.zeros((InputData.input_max_dim, InputData.input_max_dim,3),dtype=np.int32);
      blank_img[0:resized_full_img.shape[0], 0:resized_full_img.shape[1],:] = resized_full_img;
      #put the current image in the array for the entire batch 
      batch_imgs[il,:,:,:] = blank_img;

      #record the label
      gt_labels[il] = label


    #end for il, each prop box

    #update the current image index 
    self.cur_img_index = img_indices[-1] % num_imgs

    
    batch = [batch_imgs, gt_labels]
    return batch 



  #next batch 



