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
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
# Categorical likelihood for classication.
y = Categorical(tf.matmul(x,w)+b)

# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qb = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

# We use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.int32, [N])
# Define the VI inference technique, ie. minimise the KL divergence between q and p.
inference = ed.KLqp({w: qw, b: qb}, data={y:y_ph})

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
Y_test = test_data[1]

# Generate samples the posterior and store them.
n_samples = 10
prob_lst = []

for i in range(n_samples):
    w_samp = qw.sample()
    b_samp = qb.sample()

    # Also compue the probabiliy of each class for each (w,b) sample.
    prob = tf.nn.softmax(tf.matmul( X_test, w_samp ) + b_samp)
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
