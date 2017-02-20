import os
import numpy as np
from scipy import misc 
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
import random as rnd

#TODO - allow more than one scene
#     -support multiple image sizes
#     -don't read all images at start

class InputData:
  'Organizes and provides training data'

  BASE_PATH = '/playpen/ammirato/Data/RohitData/';
  META_BASE_PATH = '/playpen/ammirato/Data/RohitMetaData/';
  

  #input_max_dim = 224 
  num_cats = 2 

  
  IMG_CH = 3 


  def __init__(self,scene_name,input_max_dim,jpg_imgs=1, random=0):
    "sets scene for this object, and initializes data array"

    self.scene_name = scene_name
    self.random = random 
    self.input_max_dim = input_max_dim
    if random == 0:
      self.scene_path = os.path.join(InputData.BASE_PATH, scene_name)
      self.meta_path = os.path.join(InputData.META_BASE_PATH, scene_name)
      self.cur_img_index = 0
      self.init_data() 
    else:
      self.num_examples = 1000 
      
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


    if self.random == 0:    
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
      batch_imgs = np.zeros((len(img_indices), self.input_max_dim, 
                              self.input_max_dim,InputData.IMG_CH),dtype=np.float32)
     

      #hold ground truth labels for all images
      #gt_labels = np.zeros((cur_props.shape[0], InputData.num_cats)) 
      gt_labels = np.zeros((len(img_indices))) 


      #for each  image
      for il in range(0,len(img_indices)):
        index = img_indices[il]
        img_path = self.paths_and_labels[index][0];      
        label = int(self.paths_and_labels[index][1]);      
        full_img = misc.imread(os.path.join(img_path));    

        #print 'Image type: ' + str(full_img.dtype)
        #plt.imshow(full_img); plt.show()


        #resize the full_img, but KEEP ASPECT RATIO
        scale_factor = self.input_max_dim / float(max(full_img.shape[0:2]));
        new_size = (int(full_img.shape[0]*scale_factor), int(full_img.shape[1]*scale_factor),3)
  #      resized_full_img = misc.imresize(full_img, new_size);
        resized_full_img = cv2.resize(full_img,
                               (self.input_max_dim,self.input_max_dim))      

        #make reiszed img float instead of uint8
        resized_full_img = resized_full_img.astype(np.float32) 

        #normalize
        resized_full_img[:,:,0] =  resized_full_img[:,:,0] - 120.763;
        resized_full_img[:,:,1] =  resized_full_img[:,:,1] - 119.409;
        resized_full_img[:,:,2] =  resized_full_img[:,:,2] - 100.810;
        
        resized_full_img[:,:,0] =  resized_full_img[:,:,0] / 72.183;
        resized_full_img[:,:,1] =  resized_full_img[:,:,1] / 80.042;
        resized_full_img[:,:,2] =  resized_full_img[:,:,2] / 78.343;
        
       # resized_full_img[:,:,0] =  resized_full_img[:,:,0] / 255.0;
       # resized_full_img[:,:,1] =  resized_full_img[:,:,1] /255.0;
       # resized_full_img[:,:,2] =  resized_full_img[:,:,2] /255.0;



        #pad image with zeros to get to desired size
        #blank_img = np.zeros((self.input_max_dim, self.input_max_dim,3),dtype=np.float32);
        #blank_img[0:resized_full_img.shape[0], 0:resized_full_img.shape[1],:] = resized_full_img;
        #put the current image in the array for the entire batch 
        #batch_imgs[il,:,:,:] = blank_img;
        
        batch_imgs[il,:,:,:] = resized_full_img;
    
        #plt.imshow(resized_full_img); plt.show()

        #record the label
        gt_labels[il] = label


      #end for il, each prop box

      #update the current image index 
      self.cur_img_index = img_indices[-1]+1 % num_imgs

      
      batch = [batch_imgs, gt_labels]
      return batch 

    elif self.random == 1:
      #make images with mostly black background and a 5x5 white box
      box_size = 5

      #make an array to hold all the images of the region proposals
      batch_imgs = np.zeros((batch_size, self.input_max_dim, 
                              self.input_max_dim,1),dtype=np.float32)
     

      #hold ground truth labels for all images
      #gt_labels = np.zeros((cur_props.shape[0], InputData.num_cats)) 
      gt_labels = np.zeros((batch_size)) 

    

      for i in range(batch_size):

        gt_label = round(np.random.rand(1)[0]*1.3)
      
        #first make random black background image
        img = np.random.rand(self.input_max_dim,self.input_max_dim)*255
       
        if gt_label ==1:
          #put in the white square  
          loc = np.random.rand(2)*(self.input_max_dim-box_size)
          img[loc[0]:loc[0]+5, loc[1]:loc[1]+5] = np.random.rand(5,5)*10 + 245 

        #normalize the image so everything is between -1 and 1
        img = img -127
        img = img/127
        
        batch_imgs[i,:,:,:] = np.reshape(img,(img.shape[0],img.shape[1],1))
        gt_labels[i] = gt_label
      
        batch = [batch_imgs,gt_labels]
        return batch

  #next batch 



