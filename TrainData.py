import os
import numpy as np
from scipy import misc 
import scipy.io as sio

#TODO - allow more than one scene
#     -support multiple image sizes
#     -don't read all images at start

class TrainData:
  'Organizes and provides training data'

  BASE_PATH = '/playpen/ammirato/Data/RohitData/';
  

  input_max_dim = 300
  num_cats = 33 

  IMG_DIM_REDUCE_FACTOR = 2
  
  IMG_W = 1920/IMG_DIM_REDUCE_FACTOR
  IMG_H = 1080/IMG_DIM_REDUCE_FACTOR
  IMG_CH = 3 


  def __init__(self,scene_name,jpg_imgs=1):
    "sets scene for this object, and initializes data array"
    self.scene_name = scene_name
    self.scene_path = os.path.join(TrainData.BASE_PATH, scene_name)
    self.cur_img_index = 0

    self.init_data(jpg_imgs) 
  #init


  def init_data(self, jpg_imgs):
    "initializes data array with names of images to use"

    #get image names (jpg or png)
    img_path = ''
    if jpg_imgs:
      img_path = os.path.join(self.scene_path, 'jpg_rgb') 
    else:
      img_path = os.path.join(self.scene_path, 'rgb') 
    self.file_names = os.listdir(img_path)

    #load all the images and labels
    self.images = np.zeros((len(self.file_names), TrainData.IMG_H, 
                            TrainData.IMG_W,TrainData.IMG_CH),dtype=np.int32)

    #self.labels = [None]*len(self.file_names)
    #self.proposals = dict.fromkeys(self.file_names)

    boxes = sio.loadmat(os.path.join(self.scene_path, 'region_proposals', 
                                     'all_selected_proposals.mat'));

    self.proposals = boxes['boxes']

    counter = 0
    for fname in self.file_names:

      #load and resize image
      #img = misc.imread(os.path.join(img_path, fname))
      #img = misc.imresize(img,(TrainData.IMG_H, TrainData.IMG_W, 3))
      #self.images[counter,:,:,:] = img.astype(np.int32)

      #load labels
      #boxes = sio.loadmat(os.path.join(self.scene_path, 'labels', 
      #                               'bounding_boxes_by_image_instance', fname[0:10] + '.mat'))

      #boxes = sio.loadmat(os.path.join(self.scene_path, 'region_proposals', 
      #                               'selected_region_proposals', fname[0:10] + '.mat'))
      #boxes = boxes['boxes']
      #self.proposals[fname] = boxes;

      counter += 1
  #init_data

    

  def next_batch(self,batch_size):
    "returns the next batch_size images"

    #get total number of training images
    num_imgs = self.proposals.shape[0]

    #check to see if we need to wrap around to beginning  
    if (self.cur_img_index + batch_size) >= num_imgs :
      end_range = range(self.cur_img_index, num_imgs)
      start_range = range(0, batch_size - len(end_range)) 
      img_indices = end_range + start_range
    else:
      img_indices = range(self.cur_img_index, self.cur_img_index+batch_size)

    #get the chosen image region proposals
    cur_props = self.proposals[img_indices,:]

    #make an array to hold all the images of the region proposals
    prop_imgs = np.zeros((cur_props.shape[0], TrainData.input_max_dim, 
                            TrainData.input_max_dim,TrainData.IMG_CH),dtype=np.int32)
   

    #holds one hot vectors for labels
    gt_labels = np.zeros((cur_props.shape[0], TrainData.num_cats)) 

    #for each prop, get its image
    for il in range(0,cur_props.shape[0]):
      
      #get the box and the full rgb image
      box = cur_props[il,:]
      #full_img = self.images[box[5],:,:,:]
      img_name = str(box[5]).zfill(6) + '0101.jpg';      
      img_path = os.path.join(self.scene_path, 'jpg_rgb') 
      full_img = misc.imread(os.path.join(img_path, img_name));    

 
      #crop out the prop box 
      box_img = full_img[box[1]:box[3], box[0]:box[2],:]


      #resize the box_img, but KEEP ASPECT RATIO
      scale_factor = TrainData.input_max_dim / float(max(box_img.shape[0:2]));
      new_size = (int(box_img.shape[0]*scale_factor), int(box_img.shape[1]*scale_factor),3)


      try:
        resized_box_img = misc.imresize(box_img, new_size);
      except:
        breakp = 1




      #pad image with zeros to get to desired size
      blank_img = np.zeros((TrainData.input_max_dim, TrainData.input_max_dim,3),dtype=np.int32);
      blank_img[0:resized_box_img.shape[0], 0:resized_box_img.shape[1],:] = resized_box_img;
      #put the current image in the array for the entire batch 
      prop_imgs[il,:,:,:] = blank_img;

      #record the label
      gt_labels[il,box[4]-1] = 1 

    #end for il, each prop box

    #update the current image index 
    self.cur_img_index = img_indices[-1] % num_imgs

    
    batch = [prop_imgs, gt_labels]
    return batch 

  #next batch 


#  def next_batch(self,batch_size):
#    "returns the next batch_size images"
#
#    #get total number of training images
#    num_imgs = self.images.shape[0]
#
#    #check to see if we need to wrap around to beginning  
#    if (self.cur_img_index + batch_size) >= num_imgs :
#      end_range = range(self.cur_img_index, num_imgs)
#      start_range = range(0, batch_size - len(end_range)) 
#      img_indices = end_range + start_range
#    else:
#      img_indices = range(self.cur_img_index, self.cur_img_index+batch_size)
#
#    #get the images and update the current image index
#    img_batch = self.images[img_indices,:,:,:]
#    self.cur_img_index = img_indices[-1] % num_imgs
#
#    return img_batch

  #next batch 

 
