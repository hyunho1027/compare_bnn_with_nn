import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import Categorical, Normal
import edward as ed

def next_batch(dataset, N, i):
    left = i*N % len(dataset[0])
    right = (i+1)*N % len(dataset[0])
    
    if left < right :
        return dataset[0][left:right], dataset[1][left:right]
    else:
        return np.vstack((dataset[0][left:],dataset[0][:right])), np.vstack((dataset[1][left:],dataset[1][:right]))

train_data, test_data = tf.keras.datasets.cifar10.load_data()

# parameters
N = 256   # number of images in a minibatch.
D = 32*32*3   # number of features.
K = 10    # number of classes.

# Create a placeholder to hold the data (in minibatches) in a TensorFlow graph.
x = tf.placeholder(tf.float32, [None, D])
# Normal(0,1) priors for the variables. Note that the syntax assumes TensorFlow 1.1.
w1 = Normal(loc=tf.zeros([D, 1024]), scale=tf.ones([D, 1024]))
b1 = Normal(loc=tf.zeros(1024), scale=tf.ones(1024))
l1 = tf.nn.leaky_relu(tf.matmul(x,w1)+b1)

w2 = Normal(loc=tf.zeros([1024, 1024]), scale=tf.ones([1024, 1024]))
b2 = Normal(loc=tf.zeros(1024), scale=tf.ones(1024))
l2 = tf.nn.leaky_relu(tf.matmul(l1,w2)+b2)

w3 = Normal(loc=tf.zeros([1024, K]), scale=tf.ones([1024, K]))
b3 = Normal(loc=tf.zeros(K), scale=tf.ones(K))

# Categorical likelihood for classication.
y = Categorical(tf.matmul(l2,w3)+b3)

# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qw1 = Normal(loc=tf.Variable(tf.random_normal([D, 1024])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, 1024]))))
qb1 = Normal(loc=tf.Variable(tf.random_normal([1024])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1024]))))
qw2 = Normal(loc=tf.Variable(tf.random_normal([1024, 1024])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1024, 1024]))))
qb2 = Normal(loc=tf.Variable(tf.random_normal([1024])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1024]))))
qw3 = Normal(loc=tf.Variable(tf.random_normal([1024, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1024, K]))))
qb3 = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

# We use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.int32, [N])
# Define the VI inference technique, ie. minimise the KL divergence between q and p.
inference = ed.KLqp({w1: qw1, b1: qb1, w2: qw2, b2: qb2, w3: qw3, b3: qb3}, data={y:y_ph})

# Initialse the infernce variables
inference.initialize(n_iter= 5000, n_print=100, scale={y: float(len(train_data[0])) / N})

# We will use an interactive session.
sess = tf.InteractiveSession()
# Initialise all the vairables in the session.
tf.global_variables_initializer().run()

# Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
for i in range(inference.n_iter):
    X_batch, Y_batch = next_batch(train_data,N,i)
    info_dict = inference.update(feed_dict={x: np.reshape(X_batch,(N,32*32*3)), y_ph: np.reshape(Y_batch,(-1))})
    inference.print_progress(info_dict)

# Load the test images.
X_test = np.reshape(test_data[0],(-1,32*32*3)).astype(np.float32)
# TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
Y_test = np.reshape(test_data[1],(-1))

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
    l1_samp = tf.nn.leaky_relu(tf.matmul( X_test,w1_samp ) + b1_samp)
    l2_samp = tf.nn.leaky_relu(tf.matmul( l1_samp,w2_samp ) + b2_samp)
    l3_samp = tf.matmul( l2_samp,w3_samp ) + b3_samp
        
    prob = tf.nn.softmax(l3_samp)
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

