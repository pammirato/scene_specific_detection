import os
import numpy as np
from scipy import misc 
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf


from TrainData import *
import helper_functions as hf


#set up paths to images and region proposals
scene_path = '/playpen/ammirato/Data/RohitData/Bedroom_01_2'
model_path = './models/first_3deep.ckpt'
image_path = os.path.join(scene_path, 'jpg_rgb')
region_prop_path = os.path.join(scene_path, 'region_proposals', 'full_region_proposals')


test_data = TrainData('Bedroom_01_2')




# *********************************************************************



# *******     functions from Deep MNIST tensorflow tutorial  **********
#to init weights and bias 
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# *******     END tutorial functions **********




#define model

#Input Layer  - 300x300 rgb images
x = tf.placeholder(tf.float32, shape=[None, 300,300,3])
#one hot vector for class prediction
y_ = tf.placeholder(tf.float32, shape=[None, 33])

#First hidden layer, 32 5x5x3 conv filters, and 2x2 max pooling
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second hidden layer, 64 5x5 conv filters, and 2x2 max pooling
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Third hidden layer, fully connected with 1024 neurons
fc1_in_dim = 75*75*64;

W_fc1 = weight_variable([fc1_in_dim, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,fc1_in_dim])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Drop out for fully connected layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Output layer 
W_fc2 = weight_variable([1024, 33])
b_fc2 = bias_variable([33])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Train Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#load network
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, model_path)

# *********************************************************************











batch = test_data.next_batch(50)
acc = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})















##size of image fed to network
#img_size = 300
#
#
##get all the image names, 
#image_names = os.listdir(image_path)
#
##for each image, fed each proposal to the network
#for fname in image_names:
#
#
#  #load and resize image
#  img = misc.imread(os.path.join(image_path, fname))
#
#  #load region proposals for this image 
#  props = sio.loadmat(os.path.join(region_prop_path, fname[0:10] + '.mat'))
#  props = props['boxes']
#
#  #for each region proposal, crop the orginal image,
#  # and feed the new image to the network
#  for jl in range(0, props.shape[0]):
#
#    #get the proposed box
#    prop_box = props[jl,:]
#    #get the cropped/squared image
#    prop_img = hf.get_square_cropped_image(img, prop_box, img_size) 
#
#    
#
#
#  #for jl, each region proposal
##for fname in image names






