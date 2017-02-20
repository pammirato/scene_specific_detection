import data_processing.InputData as InputData

from scipy import misc 
import time
import tensorflow as tf
#import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import numpy as np


batch_size = 1
trained_model_dir= '/playpen/ammirato/Documents/scene_specific_detection/trained_models/'
model_name = 'test-999'
image_size = 32












def fill_feed_dict(data_set, images_pl, labels_pl, batch_size):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
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


















sess = tf.Session()




train_data = InputData.InputData('Home_14_1',32);







saver = tf.train.import_meta_graph(trained_model_dir + model_name + '.meta')
saver.restore(sess,trained_model_dir + model_name)


g = tf.get_default_graph()
logits = g.get_tensor_by_name('softmax_out:0')


# Generate placeholders for the images and labels.
images_placeholder = g.get_tensor_by_name('Placeholder:0')
#org_shape = images_placeholder.shape()
#images_placeholder = tf.reshape(images_placeholder,[None,org_shape[1],
#                                                    org_shape[2],org_shape[3]])
labels_placeholder = g.get_tensor_by_name('Placeholder_1:0')

hidden1 = g.get_tensor_by_name('conv1_h:0')
weights1 = g.get_tensor_by_name('conv1_w:0')

correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
# Return the number of true entries.
eval_correct =  tf.reduce_sum(tf.cast(correct, tf.int32))





num_filters = 3

label = 0

while label!=1:

  feed_dict = fill_feed_dict(train_data,
                             images_placeholder,
                             labels_placeholder,
                             1)


  label = feed_dict[labels_placeholder]
  assert len(label) == 1
  label = label[0]
  print label



[h1_acts,filters1]   = sess.run([hidden1,weights1], feed_dict=feed_dict)

h1_acts = h1_acts[0]

images= feed_dict[images_placeholder]

img = images[0,:,:,:]

img[:,:,0] = img[:,:,0] * 72.183
img[:,:,1] = img[:,:,1] * 80.042
img[:,:,2] = img[:,:,2] * 78.343
img[:,:,0] = img[:,:,0] + 120.763 
img[:,:,1] = img[:,:,1] + 119.409
img[:,:,2] = img[:,:,2] + 100.810




filters_fig,filters_ax = plt.subplots(filters1.shape[3])
acts_fig,acts_ax = plt.subplots(filters1.shape[3])

for i in range(filters1.shape[3]):
  filters_ax[i].imshow(filters1[:,:,:,i] )

for i in range(h1_acts.shape[2]):
  acts_ax[i].imshow(h1_acts[:,:,i])

img_fig,img_ax = plt.subplots(1)
img_ax.imshow(img.astype(np.uint8))


plt.draw()
plt.pause(.001)





