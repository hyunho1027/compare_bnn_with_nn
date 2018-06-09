import numpy as np
import tensorflow as tf

def make_val(train_data, val_size):

    val_idx = np.random.choice(range(len(train_data[0])), val_size, replace=False)
    val_data = (train_data[0][val_idx], train_data[1][val_idx])
    train_data = (np.delete(train_data[0], val_idx, 0), np.delete(train_data[1], val_idx, 0))

    return train_data, val_data

def make_onehot(dataset, cate_num):

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
X = tf.placeholder(tf.float32, [None, 32*32*3])
Y = tf.placeholder(tf.float32, [None, 10])

dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size=10000).batch(batch_size).repeat()

# create the iter
iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()

# create the model
W1 = tf.Variable(tf.random_normal([32*32*3,10], stddev=0.01))
b1 = tf.Variable(tf.random_normal([10]))

logits = tf.matmul(features, W1) + b1

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
    sess.run(iter.initializer, feed_dict={X: np.reshape(train_data[0],(-1,32*32*3)), Y: train_data[1], batch_size: BATCH_SIZE})
    total_cost = 0
    train_accu = 0

    for i in range(total_batch):
        a, c, _ = sess.run([accuracy, cost, optimizer])
        total_cost += c
        train_accu += a

    sess.run(iter.initializer, feed_dict={X: np.reshape(val_data[0],(-1,32*32*3)), Y: val_data[1], batch_size: len(val_data[0])})
    val_accu = sess.run(accuracy)

    print("Epoch: {}, Loss: {:.4f}, train Accu: {:.4f}, val Accu: {:.4f}".format(epoch + 1, total_cost / total_batch,
                                                                                 train_accu / total_batch, val_accu))
print('Learning Finished!')

# Test model and check accuracy
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: np.reshape(test_data[0],(-1,32*32*3)), Y: test_data[1]}))