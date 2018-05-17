import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal
import edward as ed

# CIFAR-10 데이터를 다운로드 받기 위한 helpder 모듈인 load_data 모듈을 임포트합니다.
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(x_train, y_train), (x_test, y_test) = load_data()

# parameters
N = 50000   # number of images in a minibatch.
D = 3072   # number of features.
K = 10    # number of classes.


# Create a placeholder to hold the data (in minibatches) in a TensorFlow graph.
x = tf.placeholder(tf.float32, [None, D])
# Normal(0,1) priors for the variables. Note that the syntax assumes TensorFlow 1.1.
w1 = Normal(loc=tf.zeros([D, 1536]), scale=tf.ones([D, 1536]))
b1 = Normal(loc=tf.zeros(1536), scale=tf.ones(1536))
l1 = tf.matmul(x,w1)+b1

w2 = Normal(loc=tf.zeros([1536, 1536]), scale=tf.ones([1536, 1536]))
b2 = Normal(loc=tf.zeros(1536), scale=tf.ones(1536))
l2 = tf.matmul(l1,w2)+b2


w3 = Normal(loc=tf.zeros([1536, K]), scale=tf.ones([1536, K]))
b3 = Normal(loc=tf.zeros(K), scale=tf.ones(K))

# Categorical likelihood for classication.
y = Categorical(tf.matmul(l2,w3)+b3)

# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qw1 = Normal(loc=tf.Variable(tf.random_normal([D, 1536])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, 1536]))))
qb1 = Normal(loc=tf.Variable(tf.random_normal([1536])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1536]))))

qw2 = Normal(loc=tf.Variable(tf.random_normal([1536, 1536])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1536, 1536]))))
qb2 = Normal(loc=tf.Variable(tf.random_normal([1536])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1536]))))

qw3 = Normal(loc=tf.Variable(tf.random_normal([1536, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1536, K]))))
qb3 = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

x_train = np.reshape(x_train,(-1,3072))
x_test = np.reshape(x_test,(-1,3072))
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

# We use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.int32, [N])
# Define the VI inference technique, ie. minimise the KL divergence between q and p.
inference = ed.KLqp({w1: qw1, b1: qb1,
                     w2: qw2, b2: qb2,
                     w3: qw3, b3: qb3,}, data={y:y_ph})

# Initialse the infernce variables
inference.initialize(n_iter=1000, n_print=1, scale={y: float(len(x_train) / N)})

# We will use an interactive session.
sess = tf.InteractiveSession()
# Initialise all the vairables in the session.
tf.global_variables_initializer().run()

# Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
for _ in range(inference.n_iter):
    X_batch, Y_batch = x_train, np.reshape(sess.run(y_train),(-1,10))
    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)


# Load the test images.
X_test = tf.convert_to_tensor(x_test,dtype=tf.float32)
# TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
Y_test = np.argmax(np.reshape(sess.run(y_test),(-1,10)),axis=1)

# Generate samples the posterior and store them.
n_samples = 10
prob_lst = []
samples1 = []
samples2 = []
samples3 = []
w1_samples = []
b1_samples = []
w2_samples = []
b2_samples = []
w3_samples = []
b3_samples = []
for i in range(n_samples):
    w1_samp = tf.convert_to_tensor(qw1.sample(),dtype=tf.float32)
    b1_samp = qb1.sample()
    w2_samp = tf.convert_to_tensor(qw2.sample(),dtype=tf.float32)
    b2_samp = qb2.sample()
    w3_samp = tf.convert_to_tensor(qw3.sample(),dtype=tf.float32)
    b3_samp = qb3.sample()
    w1_samples.append(w1_samp)
    b1_samples.append(b1_samp)
    w2_samples.append(w2_samp)
    b2_samples.append(b2_samp)
    w3_samples.append(w3_samp)
    b3_samples.append(b3_samp)
    # Also compue the probabiliy of each class for each (w,b) sample.
    prob = tf.nn.softmax(tf.matmul(tf.matmul(tf.matmul( X_test, w1_samp) + b1_samp, w2_samp) + b2_samp, w3_samp) + b3_samp)
    prob_lst.append(prob.eval())
    sample1 = tf.concat([tf.reshape(w1_samp,[-1]),b1_samp],0)
    sample2 = tf.concat([tf.reshape(w2_samp,[-1]),b2_samp],0)
    sample3 = tf.concat([tf.reshape(w3_samp,[-1]),b3_samp],0)
    samples1.append(sample1.eval())
    samples2.append(sample2.eval())
    samples3.append(sample3.eval())
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
