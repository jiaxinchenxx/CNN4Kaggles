import tensorflow as tf
from model import Model
import numpy as np

FLAGS = tf.app.flags.FLAGS
image_width = 48
image_height = 48
batch_sz = 128



def evaluate(test_data, modelpath):
    with tf.Graph().as_default():

        predictions = []

        x = tf.placeholder(shape=[None, image_width, image_height, 1], dtype=tf.float32, name='x')
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

        model = Model()

        logits = model.inference(x, keep_prob=0.0)
        predicts = model.predictions(logits)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, modelpath)

            N = test_data.shape[0]
            scale = int(np.floor(N / batch_sz)) + 1

            for bz in range(scale):
                offset = (bz * batch_sz) % (N - batch_sz)
                batch_x = test_data[offset: (offset + batch_sz)]

                pred = sess.run(predicts, feed_dict= {x: batch_x, keep_prob : 0.0})

                predictions.append(pred)

            predictions = np.hstack(predictions)

        return predictions

