import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal
import edward as ed
import pandas as pd

# Use the TensorFlow method to download and/or load the data.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
N = 128   # number of images in a minibatch.
D = 784   # number of features.
K = 10    # number of classes.


# Create a placeholder to hold the data (in minibatches) in a TensorFlow graph.
x = tf.placeholder(tf.float32, [None, D])
# Normal(0,1) priors for the variables. Note that the syntax assumes TensorFlow 1.1.
w1 = Normal(loc=tf.zeros([D, 256]), scale=tf.ones([D, 256]))
b1 = Normal(loc=tf.zeros(256), scale=tf.ones(256))
l1 = tf.nn.relu(tf.matmul(x,w1)+b1)


w2 = Normal(loc=tf.zeros([256, 256]), scale=tf.ones([256, 256]))
b2 = Normal(loc=tf.zeros(256), scale=tf.ones(256))
l2 = tf.nn.relu(tf.matmul(l1,w2)+b2)


w3 = Normal(loc=tf.zeros([256, K]), scale=tf.ones([256, K]))
b3 = Normal(loc=tf.zeros(K), scale=tf.ones(K))

# Categorical likelihood for classication.
y = Categorical(tf.matmul(l2,w3)+b3)

# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qw1 = Normal(loc=tf.Variable(tf.random_normal([D, 256])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, 256]))))
qb1 = Normal(loc=tf.Variable(tf.random_normal([256])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([256]))))

qw2 = Normal(loc=tf.Variable(tf.random_normal([256, 256])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([256, 256]))))
qb2 = Normal(loc=tf.Variable(tf.random_normal([256])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([256]))))

qw3 = Normal(loc=tf.Variable(tf.random_normal([256, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([256, K]))))
qb3 = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

# We use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.int32, [N])

# Define the VI inference technique, ie. minimise the KL divergence between q and p.
inference = ed.KLqp({w1: qw1, b1: qb1,
                     w2: qw2, b2: qb2,
                     w3: qw3, b3: qb3,}, data={y:y_ph})

# Initialse the infernce variables
inference.initialize(n_iter=10000, n_print=100, scale={y: float(mnist.train.num_examples) / N})

# We will use an interactive session.
sess = tf.InteractiveSession()
# Initialise all the vairables in the session.
tf.global_variables_initializer().run()

# Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
for _ in range(inference.n_iter):
    X_batch, Y_batch = mnist.train.next_batch(N)
    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)


# Load the test images.
X_test = mnist.test.images
# TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
Y_test = np.argmax(mnist.test.labels,axis=1)

# Generate samples the posterior and store them.
n_samples = 10
prob_lst = []
for i in range(n_samples):
    w1_samp = qw1.sample()
    b1_samp = qb1.sample()
    w2_samp = qw2.sample()
    b2_samp = qb2.sample()
    w3_samp = qw3.sample()
    b3_samp = qb3.sample()

    # Also compue the probabiliy of each class for each (w,b) sample.
    prob = tf.nn.softmax(tf.matmul(tf.matmul(tf.matmul( X_test, w1_samp) + b1_samp, w2_samp) + b2_samp, w3_samp) + b3_samp)
    prob_lst.append(prob.eval())
    
    print(i+1, "steps completed.")


# Compute the accuracy of the model. 
# For each sample we compute the predicted class and compare with the test labels.
# Predicted class is defined as the one which as maximum proability.
# We perform this test for each (w,b) in the posterior giving us a set of accuracies
# Finally we make a histogram of accuracies for the test data.
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
