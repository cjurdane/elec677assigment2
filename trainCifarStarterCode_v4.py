from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp
import os

# --------------------------------------------------
# setup
def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''
    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


ntrain = 1000 # per class
ntest =  100 # per class
nclass =  10# number of classes
imsize = 28
nchannels = 1
batchsize = 100
result_dir = './results/' # directory where the results from the training are saved
    
Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = 'CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

sess = tf.InteractiveSession()

tf_data = tf.placeholder("float", shape=[None,imsize,imsize,nchannels])
#tf variable for the data, remember shape is [None, width, height, numberOfChannels] 
tf_labels = tf.placeholder("float", shape=[None,nclass])
#tf variable for labels

# --------------------------------------------------
# model
#create your model

# first convolutional layer
W_conv1=weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
conv1 = tf.nn.conv2d(tf_data, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(conv1 + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(conv2 + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
W_fc2 = weight_variable([1024, nclass])
b_fc2 = bias_variable([nclass])
forward = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy

learningrate=1e-3
cross_entropy = -tf.reduce_sum(tf_labels*tf.log(forward))
optimizer = tf.train.AdamOptimizer(learningrate).minimize(cross_entropy)
#optimizer = tf.train.RMSPropOptimizer(learningrate).minimize(cross_entropy)
#optimizer = tf.train.GradientDescentOptimizer(learningrate).minimize(cross_entropy)
#optimizer = tf.train.AdagradOptimizer(learningrate).minimize(cross_entropy)
evaluation = tf.equal(tf.argmax(forward,1), tf.argmax(tf_labels,1))
accuracy = tf.reduce_mean(tf.cast(evaluation, "float"))



# to tf.image_summary format [batch_size, height, width, channels]
x_min = tf.reduce_min(W_conv1)
x_max = tf.reduce_max(W_conv1)
weights_0_to_1 = (W_conv1 - x_min) / (x_max - x_min)
weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)
weights_transposed = tf.transpose (weights_0_to_255_uint8, [3, 0, 1, 2])

# Add summary operations.
tf.scalar_summary(cross_entropy.op.name, cross_entropy)
tf.scalar_summary(accuracy.op.name, accuracy)
variable_summaries(W_conv1, 'Layer 1 weights')
variable_summaries(b_conv1, 'Layer 1 bias')
variable_summaries(conv1 + b_conv1, 'Layer 1 net inputs')
variable_summaries(h_conv1, 'Layer 1 activations after ReLu')
variable_summaries(h_pool1, 'Layer 1 activations after max pooling')
variable_summaries(W_conv2, 'Layer 2 weights')
variable_summaries(b_conv2, 'Layer 2 bias')
variable_summaries(conv2 + b_conv2, 'Layer 2 net inputs')
variable_summaries(h_conv2, 'Layer 2 activations after ReLu')
variable_summaries(h_pool2, 'Layer 2 activations after max pooling')
filter_summary = tf.image_summary('conv1/filters', weights_transposed, max_images=3)
# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.merge_all_summaries()
# Create a saver for writing training checkpoints.
saver = tf.train.Saver()
# Instantiate a SummaryWriter to output summaries and the Graph.
summary_writer = tf.train.SummaryWriter(result_dir, sess.graph)

# --------------------------------------------------
# optimization

sess.run(tf.initialize_all_variables())
batch_xs = np.zeros((batchsize,imsize,imsize,nchannels))
#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros((batchsize,nclass))
#setup as [batchsize, the how many classes] 

max_step=1000
train_accuracy_array, train_loss_array= [], []
for i in range(max_step): # try a small iteration size once it works then continue
    perm = np.arange(ntrain*nclass)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]
    if i%10 == 0:
        #calculate train accuracy and print it
        train_accuracy = accuracy.eval(feed_dict={tf_data:batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        #print("step %d, training accuracy %g"%(i, train_accuracy))
        train_accuracy_array.append(train_accuracy)
        train_loss=cross_entropy.eval(feed_dict={tf_data:batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        #print("step %d, training loss %g"%(i, train_loss))
        train_loss_array.append(train_loss)
        summary_str = sess.run(summary_op, feed_dict={tf_data:batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()
        #save the checkpoints every 1000 iterations
    if i % 1000 == 0 or i == max_step:
        checkpoint_file = os.path.join(result_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=i)
        #Update the events file which is used to monitor the training 
    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}) # dropout only during training

# --------------------------------------------------
# test

test_accuracy = accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
print("Test accuracy %g"%test_accuracy)
print("Learning rate %g"%learningrate)
print("Training iterations %g"%max_step)
plt.figure(1)
plt.subplot(221)
plt.title('train_accuracy')
plt.plot(train_accuracy_array)
plt.subplot(222)
plt.title('train_loss')
plt.plot(train_loss_array)
plt.show()

sess.close()