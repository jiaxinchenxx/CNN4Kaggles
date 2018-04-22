import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np

# lr 0.0005 beta1 0.89 beta2 0.995

# lr 0.0006 beta1 0.89 beta2 0.996

class Model(object):

    def __init__(self, mode = True, batch_size=36, learning_rate=0.0006, num_labels=7):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_class = num_labels
        self._mode = mode



    def inference(self, input_data, keep_prob):

        conv1 = tf.layers.conv2d(input_data, filters=64, kernel_size=[5, 5], padding='valid', \
                                 activation=tf.nn.relu, kernel_initializer = tf.keras.initializers.glorot_uniform())

        norm1 = tf.nn.lrn(conv1, 4)


        pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[3, 3], strides=2, padding= 'same')

        conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[5, 5], padding='valid', \
                                 activation=tf.nn.relu, kernel_initializer= tf.keras.initializers.glorot_uniform())


        #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size=[4, 4], padding='valid', \
                                 activation=tf.nn.relu, kernel_initializer= tf.keras.initializers.glorot_uniform())

        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)



        pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 128])

        dense = tf.layers.dense(inputs=pool3_flat, units= 1536, activation=tf.nn.relu, kernel_initializer= tf.keras.initializers.glorot_uniform())

        dropout = tf.layers.dropout(inputs=dense, rate=keep_prob, training= self._mode)

        logits = tf.layers.dense(inputs=dropout, units= self._num_class, kernel_initializer= tf.keras.initializers.glorot_uniform())

        return logits

    def train(self, loss, global_step):

        tf.summary.scalar('learning_rate', self._learning_rate)

        train_op = tf.train.AdamOptimizer(self._learning_rate, beta1= 0.89, beta2= 0.996, epsilon= 1e-8).minimize(\
          loss, global_step=global_step)


        return train_op

    def loss(self, logits, labels):

        with tf.variable_scope('loss') as scope:

            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            cost = tf.reduce_mean(cross_entropy, name=scope.name)
            tf.summary.scalar('cost', cost)

        return cost

    def accuracy(self, logits, y):
        with tf.variable_scope('accuracy') as scope:
            #accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), dtype=tf.float32),
            #                          name=scope.name)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits, 1), dtype = tf.int32), y), dtype= tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def predictions(self, logits):
        with tf.variable_scope('predictions') as scope:

            predictions = tf.cast(tf.argmax(logits, 1), dtype = tf.int32)

        return predictions

    def confusion_mat(self, predictions, y):
        with tf.variable_scope('confusion_matrix') as scope:
            cm = tf.confusion_matrix(y, predictions)
        return cm


    def F1Measurement(self, cm):
        with tf.variable_scope('F1_Measurement') as scope:
            precision = np.zeros(cm.shape[0])
            recall = np.zeros(cm.shape[0])
            normalised_cm = (cm.transpose() / np.sum(cm, axis = 1)).transpose()

            matrix4Precision = np.sum(normalised_cm, axis = 0)
            matrix4Recall = np.sum(normalised_cm, axis = 1)

            for i in range(cm.shape[0]):
                precision[i] = normalised_cm[i][i] / matrix4Precision[i] if matrix4Precision[i] != 0 else -1
                recall[i] = normalised_cm[i][i] / matrix4Recall[i] if matrix4Recall[i] !=0 else -1

            F1 = 2.0 * (precision * recall) / (precision + recall)
            return F1