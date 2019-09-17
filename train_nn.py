import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math


#directory to store the sample data
data_dir = './MNIST_DATA'
#input data set and use one hot as the encoding
mnist = input_data.read_data_sets(data_dir,one_hot = True)

n_input = 784 #input layer (28*28 pixels for each image)
n_hidden1 = 128 #1st hidden layer
n_hidden2 = 64 #2nd hidden layer
n_output = 10 #output layer (0-9 digits)

#3/5
#learning_rate = 1e-4
# variable learning rate
learning_rate = tf.placeholder(tf.float32)
# step for variable learning rate
step = tf.placeholder(tf.int32)

n_iterations = 15001
batch_size = 128

x = tf.placeholder("float",[None,n_input],name="x")

weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden2, n_output], stddev=0.1)),
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

#2/18 
#converge too slow when use sigmoid as activation funtion
#ReLu is used
#3/5
#dropout
# Probability of keeping a node during dropout 
pkeep = tf.placeholder(tf.float32,name="pkeep")

layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), biases['b1']),name='layer_1')
layer_1d = tf.nn.dropout(layer_1,pkeep)

layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1d, weights['w2']), biases['b2']),name ='layer_2')
layer_2d = tf.nn.dropout(layer_2,pkeep)
#got 'nan' when calculate cross_entropy
#y = tf.nn.softmax(tf.matmul(layer_2, weights['out']) + biases['out'])
ylogits = tf.matmul(layer_2d, weights['out']) + biases['out']
y = tf.nn.softmax(ylogits,name="y")

###train
#input the correct label
y_ = tf.placeholder("float",[None,10],name="y_")
#calculate cross-entropy = -sum(y_*log(y))
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ylogits,labels=y_))

#use Gradient descent algorithm to minimize cross-entropy
#2/18 use Adam Optimizer, it efficient to decrease the iterations from 50000 to 13000
#and converge after 37000 iterations.
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#3/5
# training step,
# the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
learning_rate = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


###evaluation
#the label 1 is the maximum in the vectors, y and y_
#return TRUE if the indexes of maximum element are the same in y and y_
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#convert TRUE to 1 and FALSE to 0, then calculate the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"),name="accuracy")
#print(sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels})) 


saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#saver.save(sess,'nn',global_step=n_iterations-1)


#train the model
for i in range(n_iterations):
	batch_xs,batch_ys = mnist.train.next_batch(batch_size)
	sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys,pkeep: 0.75, step: i})

	if i%1000 == 0:
		batch_loss,batch_accuracy = sess.run([cross_entropy,accuracy],feed_dict={x:mnist.test.images, y_:mnist.test.labels,pkeep: 1.0})
		print("Iteration",str(i), "\t| loss",str(batch_loss), "\t| accuracy",str(batch_accuracy))

saver.save(sess,'nn.ckpt')