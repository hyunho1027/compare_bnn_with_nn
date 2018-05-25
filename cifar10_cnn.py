import numpy as np
import tensorflow as tf

def make_val(train_data, val_size):
    '''
    Parameter
    train_data : train data tuple (e.g. (train_features, train_labels))
    val_num :  size of validation data set

    Return
    train_data : train data tuple (e.g. (train_features, train_labels))
    val_data : validation data tuple(e.g. (val_features, val_labels))

    Example
    train_data, val_data = make_val(train_data, 2000)
    '''
    val_idx = np.random.choice(range(len(train_data[0])), val_size, replace=False)
    val_data = (train_data[0][val_idx], train_data[1][val_idx])
    train_data = (np.delete(train_data[0], val_idx, 0), np.delete(train_data[1], val_idx, 0))

    return train_data, val_data

def make_onehot(dataset, cate_num):
    '''
    Parameter
    dataset : data tuple (e.g. (train_features, train_labels))
    cate_num : len of one-hot vector (e.g. cifar10's label: 10)

    Return
    dataset : data tuple with one-hot label

    Example
    train_data = make_onehot(train_data, 10)
    '''
    dataset = (dataset[0], np.squeeze(np.eye(cate_num)[dataset[1]]))

    return dataset


train_data, test_data = tf.keras.datasets.cifar10.load_data()

train_data, val_data = make_val(train_data, 2000)

train_data = make_onehot(train_data, 10) # Shape: ((48000, 32, 32, 3), (48000, 10))
val_data = make_onehot(val_data, 10) # Shape: ((2000, 32, 32, 3), (2000, 10))
test_data = make_onehot(test_data, 10) # Shape: ((10000, 32, 32, 3), (10000, 10))


# parameters
learning_rate = 0.001
training_epochs = 10
BATCH_SIZE = 32
batch_size = tf.placeholder(tf.int64)
total_batch = len(train_data[0]) // BATCH_SIZE

# create the Dataset
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])

dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size=10000).batch(batch_size).repeat()

# create the iter
iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()

# create the model
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(features, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])

W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)

# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + b5

# define cost/loss & optimizer & accuracy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    print('{} Epoch Start!'.format(epoch + 1))
    sess.run(iter.initializer, feed_dict={X: train_data[0], Y: train_data[1], batch_size: BATCH_SIZE})
    total_cost = 0
    train_accu = 0

    for i in range(total_batch):
        a, c, _ = sess.run([accuracy, cost, optimizer])
        total_cost += c
        train_accu += a

    sess.run(iter.initializer, feed_dict={X: val_data[0], Y: val_data[1], batch_size: len(val_data[0])})
    val_accu = sess.run(accuracy)

    print("Epoch: {}, Loss: {:.4f}, train Accu: {:.4f}, val Accu: {:.4f}".format(epoch + 1, total_cost / total_batch,
                                                                                 train_accu / total_batch, val_accu))
print('Learning Finished!')