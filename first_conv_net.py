import tensorflow as tf
from TrainData import *


#init tensorflow session and training data object
sess = tf.InteractiveSession()
train_data = TrainData('Bedroom_01_1')



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



#Input Layer  - 300x300 rgb images
x = tf.placeholder(tf.float32, shape=[None, 300,300,3])
#one hot vector for class prediction
y_ = tf.placeholder(tf.float32, shape=[None, 33])





 #First hidden layer, 32 5x5x3 conv filters, and 2x2 max pooling
num_conv1_filters = 32

W_conv1 = weight_variable([5, 5, 3, num_conv1_filters])
b_conv1 = bias_variable([num_conv1_filters])

#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#Second hidden layer, 64 5x5 conv filters, and 2x2 max pooling
num_conv2_filters = 32 

W_conv2 = weight_variable([5, 5, num_conv1_filters, num_conv2_filters])
b_conv2 = bias_variable([num_conv2_filters])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Second hidden layer, 64 5x5 conv filters, and 2x2 max pooling
num_conv3_filters = 32 

W_conv3 = weight_variable([5, 5, num_conv2_filters, num_conv3_filters])
b_conv3 = bias_variable([num_conv3_filters])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)



#Third hidden layer, fully connected with X neurons
fc1_in_dim = 38*38*num_conv3_filters

num_fc1_neurons = 2048

W_fc1 = weight_variable([fc1_in_dim, num_fc1_neurons])
b_fc1 = bias_variable([num_fc1_neurons])

h_pool3_flat = tf.reshape(h_pool3, [-1,fc1_in_dim])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)



#Drop out for fully connected layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)




#Output layer 
W_fc2 = weight_variable([num_fc1_neurons, 33])
b_fc2 = bias_variable([33])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2









#Train Model

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(500):
  batch = train_data.next_batch(50)
  if i%10 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})








#Test Model

test_data = TrainData('Bedroom_01_2')


num_imgs = test_data.get_number_of_images()
test_batch_size = 100

num_test_batches = (num_imgs / test_batch_size) + 1
test_accs = np.zeros(num_test_batches)


for il in range(0, num_test_batches):
  test_batch = test_data.next_batch(test_batch_size)
  test_accs[il] = accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})
  print('test acc: %g'%(test_accs[il]))

avg_acc = np.mean(test_accs)
print('\naverage test acc: %g'%(avg_acc))




