import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

#directory to store the sample data
data_dir = './MNIST_DATA'
#input data set
#use one hot as the encoding
#do not reshape the 28*28 matrix to 784 dimensions of vector
#mnist = input_data.read_data_sets(data_dir,one_hot = True, reshape=False)
mnist = input_data.read_data_sets(data_dir,one_hot = True)

# variable learning rate
learning_rate = tf.placeholder(tf.float32)
# step for variable learning rate
step = tf.placeholder(tf.int32)

n_iterations = 2001
batch_size = 128
n_input = 784

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
# channel = 1 due to binary image
x = tf.placeholder(tf.float32,[None,n_input],name="x")
x_image = tf.reshape(x,[-1,28,28,1])
#x = tf.placeholder(tf.float32, [None, 28, 28, 1],name="x")

#dropout
# Probability of keeping a node during dropout 
pkeep = tf.placeholder(tf.float32,name='pkeep')

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
out_1 = 6  # first convolutional layer output depth
out_2 = 16  # second convolutional layer output depth
N = 200  # fully connected layer
n_output = 10

def conv2d(x,W,b,stride=1):
	x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)  

def maxpool(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

weights = {
	'w1': tf.Variable(tf.truncated_normal([5,5,1,out_1], stddev=0.1)),# 5x5 patch, 1 input channel, out_1 output channels
	'w2': tf.Variable(tf.truncated_normal([5,5,out_1,out_2], stddev=0.1)),
	'w3': tf.Variable(tf.truncated_normal([7*7*out_2,N], stddev=0.1)),
	'w4': tf.Variable(tf.truncated_normal([N, n_output], stddev=0.1)),
}

biases = {
	'b1': tf.Variable(tf.ones([out_1])/10),
	'b2': tf.Variable(tf.ones([out_2])/10),
	'b3': tf.Variable(tf.ones([N])/10),
	'b4': tf.Variable(tf.ones([n_output])/10),
}


# The model
#convolution layer
conv1 = conv2d(x_image,weights['w1'],biases['b1'])    #output 6@28*28
#max pooling
conv1 = maxpool(conv1,k=2)  #output 6@14*14

conv2 = conv2d(conv1,weights['w2'],biases['b2'])
conv2 = maxpool(conv2,k=2)

# reshape the output from the second convolution for the fully connected layer
YY = tf.reshape(conv2, shape=[-1, 7 * 7 * out_2])
fc1 = tf.nn.relu(tf.matmul(YY, weights['w3']) + biases['b3'])
fc1d = tf.nn.dropout(fc1,pkeep)
Ylogits = tf.matmul(fc1d, weights['w4']) + biases['b4']
Y = tf.nn.softmax(Ylogits,name='y')

"""
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(x, weights['w1'], strides=[1, stride, stride, 1], padding='SAME') + biases['b1'])
Y1d = tf.nn.dropout(Y1,pkeep)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1d, weights['w2'], strides=[1, stride, stride, 1], padding='SAME') + biases['b2'])
Y2d = tf.nn.dropout(Y2,pkeep)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2d, weights['w3'], strides=[1, stride, stride, 1], padding='SAME') + biases['b3'])
Y3d = tf.nn.dropout(Y3,pkeep)
# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3d, shape=[-1, 7 * 7 * out_3])

Y4 = tf.nn.relu(tf.matmul(YY, weights['w4']) + biases['b4'])
Y4d = tf.nn.dropout(Y4,pkeep)
Ylogits = tf.matmul(Y4d, weights['w5']) + biases['b5']
Y = tf.nn.softmax(Ylogits,name='y')
"""

###train
#input the correct label
y_ = tf.placeholder(tf.float32, [None, n_output],name='y_')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels=y_))

# training step,
# the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
learning_rate = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

###evaluation
#the label 1 is the maximum in the vectors, y and y_
#return TRUE if the indexes of maximum element are the same in y and y_
correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(y_,1))
#convert TRUE to 1 and FALSE to 0, then calculate the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"),name='accuracy')
#print(sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels})) 

#for visualization
tf.summary.scalar('loss',cross_entropy)
tf.summary.scalar('accuracy',accuracy)
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

merged_write = tf.summary.FileWriter("./log",sess.graph)

#train the model
for i in range(n_iterations):
	batch_xs,batch_ys = mnist.train.next_batch(batch_size)
	sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys, step: i,pkeep: 0.75, step: i})

	if (i<1000 and i%10==0) or (i>=1000 and i%1000==0):
		summary = sess.run(merged_summary_op,feed_dict={x:mnist.test.images, y_:mnist.test.labels,pkeep: 1.0})
		log_writer = tf.summary.FileWriter("./log")
		log_writer.add_summary(summary,i)
		
saver.save(sess,'cnn.ckpt')