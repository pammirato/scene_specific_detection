import numpy as np
from scipy import misc 

def get_square_cropped_image(full_img, box, final_size, keep_aspect_ration=True):
  'Returns a cropped version of the inpur image(full_img). Crop defined by box. '
  'The return image is square with size final_size. Will keep aspect ratio of box'


  #crop out the prop box 
  box_img = full_img[box[1]:box[3], box[0]:box[2],:]

  #resize the box_img, but KEEP ASPECT RATIO
  scale_factor = final_size / float(max(box_img.shape[0:2]));
  new_size = (int(box_img.shape[0]*scale_factor), int(box_img.shape[1]*scale_factor),3)
  resized_box_img = misc.imresize(box_img, new_size);

  #pad image with zeros to get to desired size
  square_img = np.zeros((final_size, final_size,3),dtype=np.int32);
  square_img[0:resized_box_img.shape[0], 0:resized_box_img.shape[1],:] = resized_box_img;

  return square_img
