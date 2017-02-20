import data_processing.InputData as InputData

from scipy import misc 
import time
import tensorflow as tf
#import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import numpy as np




#USER OPTIONS

batch_size = 1#how many images to consider at once
#where the trained models are saved
trained_model_dir= '/playpen/ammirato/Documents/scene_specific_detection/trained_models/'
model_name = 'test-999' #the model to use





def fill_feed_dict(data_set, images_pl, labels_pl, batch_size):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, InputData.InputData()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(batch_size)

  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict



def undo_image_normalization(img):
  """
    Attempts to return a normal RGB image
    
    Tries to change image values to be [0,255]
  """
 
  #hardcoded 
  img[:,:,0] = img[:,:,0] * 72.183
  img[:,:,1] = img[:,:,1] * 80.042
  img[:,:,2] = img[:,:,2] * 78.343
  img[:,:,0] = img[:,:,0] + 120.763 
  img[:,:,1] = img[:,:,1] + 119.409
  img[:,:,2] = img[:,:,2] + 100.810

  return img








## MAIN SCRIPT


#start tensorflow session
sess = tf.Session()

#load the model, and get the graph(holds all the tensors/operations)
saver = tf.train.import_meta_graph(trained_model_dir + model_name + '.meta')
saver.restore(sess,trained_model_dir + model_name)
g = tf.get_default_graph()
#logits = g.get_tensor_by_name('softmax_out:0')

#get image and label placeholders for model
images_placeholder = g.get_tensor_by_name('Placeholder:0')
labels_placeholder = g.get_tensor_by_name('Placeholder_1:0')
#get the image shape used in this model - [batch_size,width,height,channels]
image_shape = images_placeholder.get_shape().as_list()

#define where the data is coming from, and give image size(makes square images)
data = InputData.InputData('Home_14_1',image_shape[1]);




#get the tensors we are interested in displaying
hidden1 = g.get_tensor_by_name('conv1_h:0')
weights1 = g.get_tensor_by_name('conv1_w:0')



#find an image with ground truth = 1 (hardcoded not background)
label = 0
while label!=1:
  feed_dict = fill_feed_dict(data,
                             images_placeholder,
                             labels_placeholder,
                             1)
  label = feed_dict[labels_placeholder]
  assert len(label) == 1
  label = label[0]




#run the image through the model, and get the values
#of the desired tensors
[h1_acts,filters1]   = sess.run([hidden1,weights1], feed_dict=feed_dict)

#stuff
h1_acts = h1_acts[0]

#get the image
images= feed_dict[images_placeholder]
img = undo_image_normalization(images[0,:,:,:])





## DISPLAY


#filters_fig,filters_ax = plt.subplots(filters1.shape[3])
#acts_fig,acts_ax = plt.subplots(filters1.shape[3])


num_plots = filters1.shape[3] + h1_acts.shape[2]
num_cols = 2
num_rows = num_plots/num_cols

#plot the filters and activations
tensors_fig,tensors_ax = plt.subplots(num_rows,num_cols)
for i in range(num_rows):
  for j in range(num_cols)[::2]:
    tensors_ax[i][j].imshow(filters1[:,:,:,i] )
    tensors_ax[i][j+1].imshow(h1_acts[:,:,i] )



#for i in range(filters1.shape[3]):
#  filters_ax[i].imshow(filters1[:,:,:,i] )
#
#for i in range(h1_acts.shape[2]):
#  acts_ax[i].imshow(h1_acts[:,:,i])

img_fig,img_ax = plt.subplots(1)
img_ax.imshow(img.astype(np.uint8))


plt.draw()
plt.pause(.001)





