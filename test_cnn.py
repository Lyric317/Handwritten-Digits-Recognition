import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import preprocess

data_dir = './MNIST_DATA'
mnist = input_data.read_data_sets(data_dir,one_hot = True)

sess = tf.Session()
#saver = tf.train.import_meta_graph('nn-15000.meta')
#saver.restore(sess,tf.train.latest_checkpoint('./'))
saver = tf.train.import_meta_graph('cnn.ckpt.meta')
saver.restore(sess,'cnn.ckpt')


graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y_ = graph.get_tensor_by_name("y_:0")
pkeep = graph.get_tensor_by_name("pkeep:0")
accuracy = graph.get_tensor_by_name("accuracy:0")
y = graph.get_tensor_by_name("y:0")

test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, pkeep:1.0})
print("\nAccuracy on test set:", test_accuracy)

"""

test_accuracy,y = sess.run([accuracy,y], feed_dict={x: preprocess.images, y_: preprocess.labels,pkeep:1.0})
print("\nAccuracy on test set:", test_accuracy)

with tf.Session() as sess:
    print("Recognition result:",sess.run(tf.argmax(y,1)))
"""