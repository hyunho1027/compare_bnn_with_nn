import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 데이터를 다운로드 받기 위한 helpder 모듈인 load_data 모듈을 임포트합니다.
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(x_train, y_train), (x_test, y_test) = load_data()

# parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
D = 3072   # number of features.
K = 10    # number of classes.


# input place holders
X = tf.placeholder(tf.float32, [None, D])
Y = tf.placeholder(tf.float32, [None, K])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([D, 1536]))
b1 = tf.Variable(tf.random_normal([1536]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([1536, 1536]))
b2 = tf.Variable(tf.random_normal([1536]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([1536, K]))
b3 = tf.Variable(tf.random_normal([K]))
hypothesis = tf.matmul(L2, W3) + b3

x_train = np.reshape(x_train,(-1,3072))
x_test = np.reshape(x_test,(-1,3072))
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(x_train) / batch_size)
    start_batch = epoch * total_batch
    for i in range(total_batch):
        batch_xs, batch_ys = x_train[start_batch:start_batch+total_batch,:], np.reshape(sess.run(y_train[start_batch:start_batch+total_batch,:]),(-1,10))
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: x_test, Y: np.reshape(sess.run(y_test),(-1,10))}))