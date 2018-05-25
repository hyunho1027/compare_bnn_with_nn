from edward.models import Bernoulli, Normal
from edward.util import Progbar
from scipy.misc import imsave
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data

def model(images):
    net = tf.reshape(images, [-1, 28, 28, 1])
    net = slim.conv2d(net, 20, [5,5], scope='conv1')
    net = slim.max_pool2d(net, [2,2], scope='pool1')
    net = slim.conv2d(net, 50, [5,5], scope='conv2')
    net = slim.max_pool2d(net, [2,2], scope='pool2')
    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 500, scope='fc4')
    net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
    
    return net

def generator(array, batch_size):
    """Generate batch with respect to array's first axis."""
    start = 0  # pointer to where we are in iteration
    while True:
        stop = start + batch_size
        diff = stop - array.shape[0]
        if diff <= 0:
            batch = array[start:stop]
            start += batch_size
        else:
            batch = np.concatenate((array[start:], array[:diff]))
            start = diff
        batch = batch.astype(np.float32) / 255.0  # normalize pixel intensities
        batch = np.random.binomial(1, batch)  # binarize images
        yield batch


#ed.set_seed(42)

data_dir = "/tmp/data"
out_dir = "/tmp/out"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
M = 128  # batch size during training
d = 10  # latent dimension

# DATA. MNIST batches are fed at training time.
mnist = input_data.read_data_sets(data_dir, one_hot=False)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test  = mnist.test.images
y_test  = mnist.test.labels

x_train_generator = generator(x_train, M)

x_ph = tf.placeholder(tf.int32, [M, 28 * 28])
net = model(tf.cast(x_ph, tf.float32))
net = Normal(loc=??, scale=??) ###################??

data = {x: x_ph}
inference = ed.KLqp(data)
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer)

tf.global_variables_initializer().run()

n_epoch = 100
n_iter_per_epoch = x_train.shape[0] // M
for epoch in range(1, n_epoch + 1):
    print("Epoch: {0}".format(epoch))
    avg_loss = 0.0

    pbar = Progbar(n_iter_per_epoch)
    for t in range(1, n_iter_per_epoch + 1):
        pbar.update(t)
        x_batch = next(x_train_generator)
        info_dict = inference.update(feed_dict={x_ph: x_batch})
        avg_loss += info_dict['loss']

    # Print a lower bound to the average marginal likelihood for an
    # image.
    avg_loss = avg_loss / n_iter_per_epoch
    avg_loss = avg_loss / M
    print("-log p(x) <= {:0.3f}".format(avg_loss))