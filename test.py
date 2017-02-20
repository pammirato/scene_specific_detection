import __init__

import data_processing.InputData as InputData
import model_specs.conv2_fc1 as conv2_fc1

import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
vgg = nets.vgg





# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('conv1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('conv2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('fc1', 1024, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')












def fill_feed_dict(data_set, images_pl, labels_pl):
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
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)

  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict



sess = tf.Session()


#batch_size = 10
#learning_rate = .001

train_data = InputData.InputData('Bedroom_01_1');

images,labels = train_data.next_batch(FLAGS.batch_size)

images_placeholder = tf.placeholder(tf.float32, shape=(images.shape[0],
                                                         images.shape[1], 
                                                         images.shape[2], 
                                                         images.shape[3]))


predictions = vgg.vgg_16(images,num_classes=2)



# Create a variable to track the global step.
global_step = tf.Variable(0, name='global_step', trainable=False)


init_train_vars = tf.initialize_all_variables()
sess.run(init_train_vars)




model_path= '/playpen/ammirato/Data/Tensorflow_Checkpoints/vgg_16.ckpt'
log_dir = '/playpen/ammirato/Documents/Tensorflow_logs/'

#variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
variables_to_restore = slim.get_variables_to_restore(exclude=['.*/fc6', '.*/fc7', '.*/fc8'])

saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, model_path);

#init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)




#labels_placeholder = tf.placeholder(tf.int32,shape=(10,2)) 
labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))


#loss = slim.losses.softmax_cross_entropy(predictions[0], labels_placeholder)

logits = predictions[0]
labels_placeholder = tf.to_int64(labels_placeholder)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits, labels_placeholder, name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')



all_vars = tf.trainable_variables()
train_vars = all_vars[len(all_vars)-6:len(all_vars)]




# Add a scalar summary for the snapshot loss.
#tf.scalar_summary(loss.op.name, loss)
# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
# Use the optimizer to apply the gradients that minimize the loss
# (and also increment the global step counter) as a single training step.
#train_op = optimizer.minimize(loss, global_step=global_step)

train_op = optimizer.minimize(loss, global_step=global_step, var_list=train_vars)


#make op to init trainable variables
#init_train_vars = tf.initialize_variables(train_vars.append(global_step))
#sess.run(init_train_vars)






# Fill a feed dictionary with the actual set of images and labels
# for this particular training step.
feed_dict = fill_feed_dict(train_data,
                           images_placeholder,
                           labels_placeholder)

# Run one step of the model.  The return values are the activations
# from the `train_op` (which is discarded) and the `loss` Op.  To
# inspect the values of your Ops or variables, you may include them
# in the list passed to sess.run() and the value tensors will be
# returned in the tuple from the call.
_, loss_value = sess.run([train_op, loss],
                         feed_dict=feed_dict)

