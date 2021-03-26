import tensorflow as tf
import numpy as np

from utils.data_utils import apply_dropout


mat = np.array([[0.1, 0.2]])


with tf.compat.v1.Session() as sess:

    tf.random.set_seed(seed=32)

    dropout_mat = tf.nn.dropout(mat, rate=0.4)
    print(sess.run(dropout_mat))


rand = np.random.RandomState(seed=32)

for _ in range(20):
    print(apply_dropout(mat, drop_rate=0.4, rand=rand))
