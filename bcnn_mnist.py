import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal
import edward as ed

# Use the TensorFlow method to download and/or load the data.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
N = 256   # number of images in a minibatch.
D = 784   # number of features.
K = 10    # number of classes.

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
x = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)

# Normal(0,1) priors for the variables. Note that the syntax assumes TensorFlow 1.1.
w1 = Normal(loc=tf.zeros([3, 3, 1, 32]), scale=tf.ones([3, 3, 1, 32]))
l1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
l1 = tf.nn.leaky_relu(l1)
l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w2 = Normal(loc=tf.zeros([3, 3, 32, 64]), scale=tf.ones([3, 3, 32, 64]))
l2 = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME')
l2 = tf.nn.leaky_relu(l2)
l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w3 = Normal(loc=tf.zeros([3, 3, 64, 128]), scale=tf.ones([3, 3, 64, 128]))
l3 = tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME')
l3 = tf.nn.leaky_relu(l3)
l3 = tf.nn.max_pool(l3, ksize=[1, 2, 2, 1], strides=[ 1, 2, 2, 1], padding='SAME')

l3_flat = tf.reshape(l3, [-1, 128 * 4 * 4])

w4 = Normal(loc=tf.zeros([128 * 4 * 4, 625]), scale=tf.ones([128 * 4 * 4, 625]))
b4 = Normal(loc=tf.zeros(625), scale=tf.ones(625))
l4 = tf.matmul(l3_flat, w4) + b4
l4 = tf.nn.leaky_relu(l4)

w5 = Normal(loc=tf.zeros([625, 10]), scale=tf.ones([625, 10]))
b5 = Normal(loc=tf.zeros(10), scale=tf.ones(10))

# Categorical likelihood for classication.
y = Categorical(tf.matmul(l4,w5)+b5)

# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qw1 = Normal(loc=tf.Variable(tf.random_normal([3, 3, 1, 32])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([3, 3, 1, 32]))))
qw2 = Normal(loc=tf.Variable(tf.random_normal([3, 3, 32, 64])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([3, 3, 32, 64]))))
qw3 = Normal(loc=tf.Variable(tf.random_normal([3, 3, 64, 128])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([3, 3, 64, 128]))))
qw4 = Normal(loc=tf.Variable(tf.random_normal([128 * 4 * 4, 625])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([128 * 4 * 4, 625]))))       
qw5 = Normal(loc=tf.Variable(tf.random_normal([625, 10])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([625, 10]))))
qb4 = Normal(loc=tf.Variable(tf.random_normal([625])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([625]))))
qb5 = Normal(loc=tf.Variable(tf.random_normal([10])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([10]))))

# We use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.int32, [N])

# Define the VI inference technique, ie. minimise the KL divergence between q and p.
inference = ed.KLqp({w1: qw1, w2: qw2, w3: qw3, w4: qw4, w5: qw5,
                     b4: qb4, b5: qb5 }, data={y:y_ph})

# Initialse the infernce variables
inference.initialize(n_iter=5000, n_print=100, scale={y: float(mnist.train.num_examples) / N})

# We will use an interactive session.
sess = tf.InteractiveSession()
# Initialise all the vairables in the session.
tf.global_variables_initializer().run()

# Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
for _ in range(inference.n_iter):
    X_batch, Y_batch = mnist.train.next_batch(N)
    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict = inference.update(feed_dict={X: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)


# Load the test images.
X_test = np.reshape(mnist.test.images,(-1,28,28,1))
# TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
Y_test = np.argmax(mnist.test.labels,axis=1)

# Generate samples the posterior and store them.
n_samples = 5
prob_lst = []
for i in range(n_samples):
    w1_samp = tf.convert_to_tensor(qw1.sample(),dtype=tf.float32)
    w2_samp = tf.convert_to_tensor(qw2.sample(),dtype=tf.float32)
    w3_samp = tf.convert_to_tensor(qw3.sample(),dtype=tf.float32)
    w4_samp = tf.convert_to_tensor(qw4.sample(),dtype=tf.float32)
    w5_samp = tf.convert_to_tensor(qw5.sample(),dtype=tf.float32)
    b4_samp = qb4.sample()
    b5_samp = qb5.sample()
    # Also compue the probabiliy of each class for each (w,b) sample.

    l1_samp = tf.nn.conv2d(X_test, w1_samp, strides=[1, 1, 1, 1], padding='SAME')
    l1_samp = tf.nn.leaky_relu(l1_samp)
    l1_samp = tf.nn.max_pool(l1_samp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    l2_samp = tf.nn.conv2d(l1_samp, w2_samp, strides=[1, 1, 1, 1], padding='SAME')
    l2_samp = tf.nn.leaky_relu(l2_samp)
    l2_samp = tf.nn.max_pool(l2_samp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    l3_samp = tf.nn.conv2d(l2_samp, w3_samp, strides=[1, 1, 1, 1], padding='SAME')
    l3_samp = tf.nn.leaky_relu(l3_samp)
    l3_samp = tf.nn.max_pool(l3_samp, ksize=[1, 2, 2, 1], strides=[ 1, 2, 2, 1], padding='SAME')

    l3_flat_samp = tf.reshape(l3_samp, [-1, 128 * 4 * 4])

    l4_samp = tf.matmul(l3_flat_samp, w4_samp) + b4_samp
    l4_samp = tf.nn.leaky_relu(l4_samp)

    l5_samp = tf.matmul(l4_samp,w5_samp)+b5_samp

    prob = tf.nn.softmax(l5_samp)
    prob_lst.append(prob.eval())
    print(i+1, "steps completed.")


accy_test = []
for prob in prob_lst:
    y_trn_prd = np.argmax(prob,axis=1).astype(np.float32)
    acc = (y_trn_prd == Y_test).mean()
    accy_test.append(acc)

print("Expactation of Accuracy: ", sess.run(tf.reduce_mean(accy_test)))

plt.hist(accy_test)
plt.title("Histogram of prediction accuracies in the MNIST test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.show()
