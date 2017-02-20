import __init__

import tensorflow as tf
import numpy as np








def weight_variable(shape, name=""):
  """ Initialize weight matrix of given shape, drawn from normal distrubtion with std=1"""
  initial = tf.truncated_normal(shape, stddev=0.1)
  if name == "":
    return tf.Variable(initial)
  else:
    return tf.Variable(initial, name=name)
def bias_variable(shape,name=""):
  """ Initialize bias matrix of given shape with value .1"""
  initial = tf.constant(0.1, shape=shape)
  if name == "":
    return tf.Variable(initial)
  else:
    return tf.Variable(initial, name=name)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')








def inference(num_classes, images, num_c1=8, num_c2=8, num_fc1=1024):
  """ Builds network with 3 hidden layers.  2 convolutional layers and 1 fully 
    connected layer to classify a specified number of classes. 

    Last layer is fully connected softmax
    
    Args:
    num_classes: the number of classes for the model to discriminate between
    images: Image placeholder, should be batch_size x IMAGE_SIZE x IMAGE_SIZE x 3 (rgb)
    num_c1: number of convolution filters in the first convultional layer
    num_c2: same as num_c1 but for second conv layer
    num_fc1: number of neurons in fully connected layer
  """


  image_size = images.get_shape()[1].value;
  image_channels = images.get_shape()[3].value
  #first conv layer
  weights_c1 =weight_variable([5, 5, image_channels,num_c1], 'conv1_w')

 
  biases_c1 = bias_variable([num_c1], 'conv1_b')
  hidden_c1 = tf.nn.relu(conv2d(images, weights_c1) + biases_c1, name='conv1_h')
  pool_c1 =  max_pool_2x2(hidden_c1)

  image_size = image_size - 4;
  image_size = int(image_size/2); 

  #second conv layer
  weights_c2 = weight_variable([5, 5, num_c1, num_c2])
  biases_c2 = bias_variable([num_c2])
  hidden_c2 = tf.nn.relu(conv2d(pool_c1, weights_c2) + biases_c2)
  pool_c2 = max_pool_2x2(hidden_c2)
  
  image_size = image_size - 4;
  image_size = int(image_size/2); 

  #fully connected layer
  fc1_in_dim = image_size*image_size*num_c2  #4 because 2 2x2 pools

  weights_fc1 = weight_variable([fc1_in_dim, num_fc1])
  biases_fc1 = bias_variable([num_fc1])
  flatten_fc1 = tf.reshape(pool_c2, [-1,fc1_in_dim])
  hidden_fc1 = tf.nn.relu(tf.matmul(flatten_fc1, weights_fc1) + biases_fc1)


  #Drop out for fully connected layer
  #keep_prob = tf.placeholder(tf.float32)
  #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



  #output layer
  weights_fc2 = weight_variable([num_fc1, num_classes])
  biases_fc2 = bias_variable([num_classes])
  hidden_fc2 = tf.matmul(hidden_fc1, weights_fc2) + biases_fc2
  softmax_fc2 = tf.nn.softmax(hidden_fc2, name='softmax_out');


#  #output layer
##  weights_fc2 = weight_variable([image_size*image_size*3, num_classes])
##  biases_fc2 = bias_variable([num_classes])
##  flatten_fc1 = tf.reshape(images, [-1,image_size*image_size*3])
##  hidden_fc2 = tf.matmul(flatten_fc1, weights_fc2) + biases_fc2
##  softmax_fc2 = tf.nn.softmax(hidden_fc2, name='softmax_out');




#  #fully connected layer
#  fc1_in_dim = image_size*image_size*3 
#  weights_fc1 = weight_variable([fc1_in_dim, fc1_in_dim])
#  biases_fc1 = bias_variable([fc1_in_dim])
#  flatten_fc1 = tf.reshape(images, [-1,fc1_in_dim])
#  hidden_fc1 = tf.nn.relu(tf.matmul(flatten_fc1, weights_fc1) + biases_fc1)
#
#  #fully connected layer
#  fc2_in_dim = image_size*image_size
#  weights_fc2 = weight_variable([fc1_in_dim, fc2_in_dim])
#  biases_fc2 = bias_variable([fc2_in_dim])
#  hidden_fc2 = tf.nn.relu(tf.matmul(hidden_fc1, weights_fc2) + biases_fc2)
#
#  #fully connected layer
#  weights_fc3 = weight_variable([fc2_in_dim, num_classes])
#  biases_fc3 = bias_variable([num_classes])
#  hidden_fc3 = tf.matmul(hidden_fc2, weights_fc3) + biases_fc3
#  softmax_fc3 = tf.nn.softmax(hidden_fc3, name='softmax_out');
#  



  #return last layer
  #return softmax_fc2
  return softmax_fc2 
  #return hidden_fc2
#end inference




def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """

  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss






def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  #tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op





def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return (correct, tf.reduce_sum(tf.cast(correct, tf.int32)))







