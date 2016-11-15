import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#call mnist function

learningRate = 0.001
trainingIters = 100000
batchSize = 64
displayStep = 10

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 256 #number of neurons for the RNN 64 128 256
nClasses = 10#this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(0, nSteps, x) #configuring so you can get it as needed for the 28 pixels

	gruCell = rnn_cell.GRUCell(nHidden)
	#rnnCell = rnn_cell.BasicRNNCell(nHidden)
	#lstmCell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0) 
	#outputs, states = rnn.rnn(rnnCell, x, dtype=tf.float32)
	#outputs, states = rnn.rnn(lstmCell, x, dtype=tf.float32)
	outputs, states = rnn.rnn(gruCell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.initialize_all_variables()
train_accuracy_array, train_loss_array= [], []
with tf.Session() as sess:
	sess.run(init)
	step = 1

	while step* batchSize < trainingIters:
		batchX, batchY = mnist.train.next_batch(batchSize)
		batchX = batchX.reshape((batchSize, nSteps, nInput))

		sess.run(optimizer, feed_dict={x: batchX, y: batchY})

		if step % displayStep == 0:
			acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
			loss = sess.run(cost, feed_dict={x: batchX, y: batchY})
			train_accuracy_array.append(acc)
			train_loss_array.append(loss)
			print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
		step +=1
	print('Optimization finished')
	test_len = 128
	testData = mnist.test.images[:test_len].reshape((-1, nSteps, nInput))
	testLabel = mnist.test.labels[:test_len]
	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
plt.title('GRU 256 hidden units train_accuracy')
plt.plot(train_accuracy_array)
plt.show()
plt.title('GRU 256 hidden units train_loss')
plt.plot(train_loss_array)
plt.show()