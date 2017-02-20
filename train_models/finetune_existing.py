import __init__

import data_processing.InputData as InputData
import model_specs.conv2_fc1 as conv2_fc1

from scipy import misc 
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
vgg = nets.vgg





# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
flags.DEFINE_integer('max_steps', 30000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
#flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')












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
  images_feed, labels_feed, paths_feed = data_set.next_batch(FLAGS.batch_size)

  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return [feed_dict, paths_feed]








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







def do_eval(sess,
            eval_correct,
            logits,
            list_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    #print 'Eval Step: ' + str(step)
    feed_dict,_ = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    count = sess.run([eval_correct], feed_dict=feed_dict)
    true_count += count[0]
  precision = true_count / float(num_examples)
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))






def output_wrong(sess,
            eval_correct,
            logits,
            list_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  wrong_paths = []
  gt_labels = []
  out_labels = []
  for step in xrange(steps_per_epoch):
    #print 'Eval Step: ' + str(step)
    feed_dict, all_paths = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    is_correct, output, count = sess.run([list_correct, logits, eval_correct], feed_dict=feed_dict)
    true_count += count
    #get the paths of the images that were missclassified
    all_gt_labels = feed_dict[labels_placeholder]
    all_output_labels = []
    for i in range(output.shape[0]): all_output_labels.append(output[i].argmax())
    wrong_inds = [inds for inds, x in enumerate(is_correct) if not(x)]
    for i in wrong_inds:
      wrong_paths.append(all_paths[i]) 
      gt_labels.append(int(all_gt_labels[i]))
      out_labels.append(all_output_labels[i]) 

  save_path = '/playpen/ammirato/Data/RohitMetaData/Home_14_1/classification/wrong_images/'
  counter = 0
  for cur_path in wrong_paths:

    full_img = misc.imread(cur_path)    
    
    misc.imsave(save_path + str(counter) + '_gt-' + str(gt_labels[counter])
                 + '_out-' + str(out_labels[counter]) + '.jpg',full_img)   
    counter += 1

























sess = tf.Session()



train_data = InputData.InputData('Home_14_1',224);

images,labels,_ = train_data.next_batch(FLAGS.batch_size)

images_placeholder = tf.placeholder(tf.float32, shape=(images.shape[0],
                                                         images.shape[1], 
                                                         images.shape[2], 
                                                         images.shape[3]))


predictions = vgg.vgg_16(images,num_classes=2)



# Create a variable to track the global step.
global_step = tf.Variable(0, name='global_step', trainable=False)


#init_train_vars = tf.initialize_all_variables()
#sess.run(init_train_vars)




#model_path= '/playpen/ammirato/Data/Tensorflow_Checkpoints/vgg_16.ckpt'
model_path= '/playpen/ammirato/Data/Tensorflow_Checkpoints/'
log_dir = '/playpen/ammirato/Documents/Tensorflow_logs/'

##variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
#variables_to_restore = slim.get_variables_to_restore(exclude=['.*/fc6', '.*/fc7', '.*/fc8'])
#
#save_path = tf.train.latest_checkpoint(model_path)
#
#saver = tf.train.Saver(variables_to_restore)
#saver.restore(sess, model_path);
#
##init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)


variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8'])
#variables_to_restore = slim.get_variables_to_restore()
init_assign_op, init_feed_dict = slim.assign_from_checkpoint(model_path + 'vgg_16.ckpt', 
                                                              variables_to_restore)
sess.run(init_assign_op, init_feed_dict)
#
all_vars = tf.global_variables();
init_vars_op = tf.initialize_variables(all_vars[-3:-1])
sess.run(init_vars_op)


#all_vars = tf.global_variables();
#init_vars_op = tf.initialize_variables(all_vars)
#sess.run(init_vars_op)


initial_values = []

for v in all_vars:
  initial_values.append(sess.run(v))





#labels_placeholder = tf.placeholder(tf.int32,shape=(10,2)) 
labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))


#loss = slim.losses.softmax_cross_entropy(predictions[0], labels_placeholder)

#logits = predictions[0]
logits = tf.nn.softmax(predictions[0],name='softmax_out')
labels_placeholder = tf.to_int64(labels_placeholder)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits, labels_placeholder, name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')


list_correct, eval_correct = evaluation(logits,labels_placeholder)


all_vars = tf.trainable_variables()
train_vars = all_vars[len(all_vars)-6:len(all_vars)]





with tf.device('/gpu:0'):
  # Add a scalar summary for the snapshot loss.
  #tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  #train_op = optimizer.minimize(loss, global_step=global_step)

  #train_op = optimizer.minimize(loss, global_step=global_step, var_list=train_vars)
  train_op = optimizer.minimize(loss, global_step=global_step)


#make op to init trainable variables
#init_train_vars = tf.initialize_variables(train_vars.append(global_step))
#sess.run(init_train_vars)



for step in range(FLAGS.max_steps):

  start_time = time.time()

  # Fill a feed dictionary with the actual set of images and labels
  # for this particular training step.
  feed_dict,_ = fill_feed_dict(train_data,
                             images_placeholder,
                             labels_placeholder)

  # Run one step of the model.  The return values are the activations
  # from the `train_op` (which is discarded) and the `loss` Op.  To
  # inspect the values of your Ops or variables, you may include them
  # in the list passed to sess.run() and the value tensors will be
  # returned in the tuple from the call.
  _, loss_value = sess.run([train_op, loss],
                           feed_dict=feed_dict)


  duration = time.time() - start_time


  if step % 10 == 0:
    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
    counter = 0
    #for v in all_vars:
    #  new_vals = sess.run(v)
    #  print('%s max diff: %s' % (v.name, abs(new_vals - initial_values[counter]).max())) 
    #  counter += 1



  if (step +1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
#        checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
#        saver.save(sess, checkpoint_file, global_step=step)
    # Evaluate against the training set.
    print('Training Data Eval:')
    do_eval(sess,
            eval_correct,
            logits,
            list_correct,
            images_placeholder,
            labels_placeholder,
            train_data)

#    output_wrong(sess,
#            eval_correct,
#            logits,
#            list_correct,
#            images_placeholder,
#            labels_placeholder,
#            train_data)




print 'Done!'

