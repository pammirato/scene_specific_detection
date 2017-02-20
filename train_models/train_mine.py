import __init__

import data_processing.InputData as InputData
import model_specs.conv2_fc1 as model_spec


import time
import tensorflow as tf
from scipy import misc 
import os

#for debugging
import matplotlib.pyplot as plt


# Basic model parameters as external flags.
#flags = tf.app.flags
#FLAGS = flags.FLAGS
#flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
#flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
#flags.DEFINE_integer('conv1', 2, 'Number of units in hidden layer 1.')
#flags.DEFINE_integer('conv2', 2, 'Number of units in hidden layer 2.')
#flags.DEFINE_integer('fc1', 1024, 'Number of units in hidden layer 2.')
#flags.DEFINE_integer('batch_size', 32, 'Batch size.  '
#                     'Must divide evenly into the dataset sizes.')
#flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
#flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
#                     'for unit testing.')

learning_rate = .001
max_steps = 1000 #Number of training steps
batch_size = 32
image_shape = [32,32,3]
summary_dir = './summaries'
save_model_dir = './trained_models/'
model_name = 'test'



def placeholder_inputs():
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         image_shape[0], 
                                                         image_shape[1],
                                                         image_shape[2] 
                                                         ))
  labels_placeholder = tf.placeholder(tf.int32, shape=(None))
  return images_placeholder, labels_placeholder




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
  images_feed, labels_feed = data_set.next_batch(batch_size)

  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict



def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate
    batch_size:  How many images to sample at once
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // batch_size
  num_examples = steps_per_epoch * batch_size
  for step in xrange(steps_per_epoch):
    #print 'Eval Step: ' + str(step)
    feed_dict = fill_feed_dict(data_set,
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
  steps_per_epoch = data_set.num_examples // batch_size
  num_examples = steps_per_epoch * batch_size
  wrong_paths = []
  gt_labels = []
  out_labels = []
  for step in xrange(steps_per_epoch):
    #print 'Eval Step: ' + str(step)
    feed_dict = fill_feed_dict(data_set,
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
 

  #precision = true_count / num_examples
  #print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
  #      (num_examples, true_count, precision))















def run_training():
  """Train for a number of steps."""
 
  train_data = InputData.InputData('Home_14_2',image_shape[0]);
  test_data = InputData.InputData('Home_14_1',image_shape[0]);


  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs()

    # Build a Graph that computes predictions from the inference model.
    logits = model_spec.inference(InputData.InputData.num_cats, images_placeholder,
                             8,
                             8,
                             1024)

    # Add to the Graph the Ops for loss calculation.
    loss = model_spec.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = model_spec.training(loss, learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    list_correct, eval_correct = model_spec.evaluation(logits, labels_placeholder)
    
    tf.summary.scalar('loss', loss)

    tf.summary.image('input_image',images_placeholder,max_outputs=3)

    #first_filter = tf.get_default_graph().get_tensor_by_name('conv1_w:0')
    #f1 = tf.transpose(first_filter, [3,0,1,2])
    #tf.summary.image('conv1',f1,max_outputs=3)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in xrange(max_steps):
      start_time = time.time()

      #print step
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


      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 10 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step +1) % 3000 == 0 or (step + 1) == max_steps:
#        checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
#        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                train_data)
      #if (step + 1) == FLAGS.max_steps:
        #output_wrong(sess,
        #        eval_correct,
        #        logits,
        #        list_correct,
        #        images_placeholder,
        #        labels_placeholder,
        #        train_data)
        print('Testing Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                test_data)
#        # Evaluate against the validation set.
#        print('Validation Data Eval:')
#        do_eval(sess,
#                eval_correct,
#                images_placeholder,
#                labels_placeholder,
#                data_sets.validation)
#        # Evaluate against the test set.
#        print('Test Data Eval:')
#        do_eval(sess,
#                eval_correct,
#                images_placeholder,
#                labels_placeholder,
#                data_sets.test)
    #end training loop



 
    checkpoint_file = os.path.join(save_model_dir, model_name)
    saver.save(sess, checkpoint_file, global_step=step)
     
    x=1











def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
