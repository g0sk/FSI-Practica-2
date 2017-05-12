import gzip
import cPickle
import sys
import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
train_y= one_hot(train_y,10)

valuex, valuey = valid_set
valuey = one_hot(valuey,10)

test_x, test_y = test_set
test_y = one_hot(test_y, 10)

x = tf.placeholder("float", [None, 784])  #muestras
y_ = tf.placeholder("float", [None, 10])  # etiquetas

W1 = tf.Variable(np.float32(np.random.rand(784, 4)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(4)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(4, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

#h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
h = (tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

#loss = tf.reduce_sum(tf.square(y_ - y))

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

batch_size = 800
num1= sys.maxint
num2=0
epoch=0
while  epoch < 60 or num2>num1+num1*0.001 or num2< num1-num1*0.001:
    num1=num2
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    num2=sess.run(loss, feed_dict={x: valuex, y_: valuey})
    print "Epoch #:", epoch, "Error: ", num2
    epoch+=1



print "Test hechos"
nerrores=0
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):
    if np.argmax(b) != np.argmax(r):
        # print "-----> ERROR"
        nerrores += 1
print "El numero de errores es: ",nerrores


# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print train_y[57]


# TODO: the neural net!!
