import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import numpy as np

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

epochs = 10
batch_size = 128

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(epochs):
		X_train, y_train = shuffle(X_train, y_train)
		for offset in range(0, len(X_train), batch_size):
			end = offset + batch_size
			X_batch, y_batch = X_train[offset:end], y_train[offset:end]
			sess.run(train_op, feed_dict = {x: X_batch, y: y_batch})
		current_cost = sess.run(cost, feed_dict = {x: X_train, y: y_train})
		val_accuracy = sess.run(accuracy_op, feed_dict = {x: X_val, y: y_val})
		print('Cost at epoch {0} is {0}'.format(epoch, current_cost))
		print('Cost validation accuracy is {0}'.format(val_accuracy))






