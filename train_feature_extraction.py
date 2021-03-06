import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import numpy as np

epochs = 10
batch_size = 128

# TODO: Load traffic signs data.
traffic_sign_data = pickle.load(open("train.p", "rb" ))

train_data = traffic_sign_data['features']
train_label = traffic_sign_data['labels']

# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.33, random_state=0)

nb_classes = len(np.unique(y_train))

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, shape = (None, 32, 32, 3))
y = tf.placeholder(tf.int64, shape = None)
one_hot_y = tf.one_hot(y, nb_classes)
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
wc8_shape = fc7.get_shape().as_list()[-1]
wc8 = tf.Variable(tf.truncated_normal(shape = (wc8_shape, nb_classes), mean = 0, stddev = 0.01))
bc8 = tf.Variable(tf.zeros(nb_classes))
logits = tf.add(tf.matmul(fc7, wc8), bc8)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
cost = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(cost, var_list=[wc8, bc8])
correct_preds = tf.equal(tf.arg_max(logits, 1), tf.arg_max(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

# TODO: Train and evaluate the feature extraction model.
init = tf.global_variables_initializer()

def eval_on_data(X, Y, sess):
	total_acc = 0
	total_loss = 0
	for offset in range(0, X.shape[0], batch_size):
		end = offset + batch_size
		X_batch = X[offset:end]
		y_batch = Y[offset:end]
		loss, acc = sess.run([cost, accuracy_op], feed_dict={x: X_batch, y: y_batch})
		total_loss += (loss * X_batch.shape[0])
		total_acc += (acc * X_batch.shape[0])
	return total_loss/X.shape[0], total_acc/X.shape[0]


with tf.Session() as sess:
	sess.run(init)
	for epoch in range(epochs):
		X_train, y_train = shuffle(X_train, y_train)
		t0 = time.time()
		for offset in range(0, len(X_train), batch_size):
			end = offset + batch_size
			X_batch, y_batch = X_train[offset:end], y_train[offset:end]
			sess.run(train_op, feed_dict = {x: X_batch, y: y_batch})
		val_loss, val_acc = eval_on_data(X_val, y_val, sess)
		print("Epoch", epoch+1)
		print("Time: %.3f seconds" % (time.time() - t0))
		print("Validation Loss =", val_loss)
		print("Validation Accuracy =", val_acc)
		print("")






