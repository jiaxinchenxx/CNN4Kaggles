import tensorflow as tf

from model import Model
import data_utils as dl
import numpy as np

FLAGS = tf.app.flags.FLAGS
NUM_LABELS = 7

image_width = 48
image_height = 48

PRINT_EVERY = 50

NUM_LABELS = 7

def train():
    model = Model()

    with tf.Graph().as_default():
        x_train, y_train, x_val, y_val = dl.loadTrainData('DATA_NEW.pkl')  # here you could load your own data, use data_utils.py

        #x_train, y_train = dl.loadWholeData('D:\CourseWork\CNN\DATA_NEW.pkl')

        #x_val, y_val = dl.loadTestData()

        x = tf.placeholder(shape=[None, image_width, image_height, 1], dtype=tf.float32, name='x')
        y = tf.placeholder(shape=[None], dtype=tf.int32, name='y')

        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

        global_step = tf.train.get_global_step()

        logits = model.inference(x, keep_prob=keep_prob)
        loss = model.loss(logits=logits, labels=y)

        accuracy = model.accuracy(logits, y)
        cm = model.confusion_mat(tf.argmax(logits, axis=1), y)

        summary_op = tf.summary.merge_all()
        train_op = model.train(loss, global_step=global_step)


        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        best = 0.0

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            sess.run(init)

            bs = FLAGS.batch_size
            N = x_train.shape[0]

            for ep in range(FLAGS.epoch):

                mask = np.random.permutation(N)
                x_train = x_train[mask]
                y_train = y_train[mask]

                scale = int(np.floor(N / bs)) + 1

                for bz in range(scale):
                    offset = (bz * FLAGS.batch_size) % (N - FLAGS.batch_size)
                    batch_x, batch_y = x_train[offset:(offset + FLAGS.batch_size)], y_train[
                                                                                offset:(offset + FLAGS.batch_size)]

                    _, cur_loss, summary = sess.run([train_op, loss, summary_op],
                                                feed_dict={x: batch_x, y: batch_y, keep_prob: 0.6})

                    if bz % PRINT_EVERY == 0:
                        validation_accuracy = accuracy.eval(feed_dict= {x: x_val, y: y_val, keep_prob : 0.0})
                        if validation_accuracy > 0.62 and validation_accuracy > best:
                            print ('best')
                            best = validation_accuracy
                            saver.save(sess, 'D:\CourseWork\CNN\\CNNwFER2013\\model_best_submit.ckpt')
                        train_accuracy = accuracy.eval(feed_dict= {x: batch_x, y: batch_y, keep_prob : 0.0})
                        print('Epoch {}/{} Iter {} Val Accuracy: {}, Train Accuracy: {}'\
                              .format(ep + 1, FLAGS.epoch, bz , ('%.5f' % validation_accuracy), ('%.5f' % train_accuracy)))


                cmatrix = cm.eval(feed_dict= {x: x_val, y: y_val, keep_prob : 0.0})
                print ('---------------------------------------------------')
                print('Ep {}/{} Confusion Matrix:\n {}'.format(ep + 1, FLAGS.epoch, cmatrix))
                print ('---------------------------------------------------')
                F1 = model.F1Measurement(cmatrix)
                print('Ep {}/{} F1 Per Class:\n {}'.format(ep + 1, FLAGS.epoch, F1))
                print('---------------------------------------------------')
                validation_accuracy = accuracy.eval(feed_dict={x: x_val, y: y_val, keep_prob: 0.0})
                print('Ep {}/{} Val Accuracy {}:'.format(ep + 1, FLAGS.epoch, ('%.5f' % validation_accuracy)))
                print('---------------------------------------------------')

            '''
            x_test, y_test = dl.loadTestData()
            test_accuracy = accuracy.eval(feed_dict={x: x_test, y: y_test, keep_prob : 0.0})
            print('---------------------------------------------------')
            print ('Final TEST ACCURACY: {}'.format(test_accuracy))
            cmatrix = cm.eval(feed_dict={x: x_test, y: y_test, keep_prob: 0.0})
            print('---------------------------------------------------')
            print('TEST Confusion Matrix:\n {}'.format(cmatrix))
            print('---------------------------------------------------')
            F1 = model.F1Measurement(cmatrix)
            print('TEST F1 Per Class:\n {}'.format(F1))
            print('---------------------------------------------------')
            '''

def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('epoch', 20, 'number of epochs')
    tf.app.flags.DEFINE_integer('batch_size', 64, 'size of training batches')
    #tf.app.flags.DEFINE_integer('num_iter', 10000, 'number of training iterations')
    #tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-10000', 'path to checkpoint file')
    #tf.app.flags.DEFINE_string('train_data', 'data/mnist_train.csv', 'path to train and test data')
    tf.app.flags.DEFINE_string('summary_dir', 'graphs', 'path to directory for storing summaries')

    tf.app.run()