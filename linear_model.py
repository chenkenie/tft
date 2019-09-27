#!/usr/bin/env python

import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

img_h = img_w = 28
img_size_flat = img_h * img_w
n_classes = 10

def load_data_old(mode='train'):
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    if mode == 'train':
        x_train, y_train, x_valid, y_valid = mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels
        return x_train, y_train, x_valid, y_valid
    else:
        x_test, y_test = mnist.test.images, mnist.test.labels
        return x_test, y_test

def load_data(mode='train'):
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_classes = 10
    
    return x_train, to_categorical(y_train), x_test, to_categorical(y_test)

    
def randomize(x, y):
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]    
    return x_batch, y_batch

def Dense(x, x_dim, y_dim, name, reuse=None):

    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable(name='weight', shape=(x_dim, y_dim), dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='bias', shape=(y_dim,), dtype=tf.float32, initializer=tf.initializers.zeros())
        y = tf.add(tf.matmul(x, w), b)

    return y
    

class TensorFlowModelTemplate:
    def __init__(self, config):
        self.config = config
        self.init_global_step()
        self.init_cur_epoch()

    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")        

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restor(sess, latest_checkpoint)
            print("Model loaded")

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.get_variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor+1)
            
    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.get_variable(0, trainable=False, name='global_step')

    def init_saver(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

class LinearModel(TensorFlowModelTemplate):
    def __init__(self, config):
        super(LinearModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def init_saver(self):
        self.saver = tr.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        raise NotImplementedError
        
def runModel(args):
    x_train, y_train, x_valid, y_valid = load_data()

    x_train = x_train.reshape((-1, img_size_flat))
    x_valid = x_valid.reshape((-1, img_size_flat))
    
    print("Shape of Input:")
    print("- Training-set:\t\t{}, {}".format(x_train.shape, y_train.shape))
    print("- Validation-set:\t{}, {}".format(x_valid.shape, y_valid.shape))    

    #define model
    # Input
    X = tf.placeholder(tf.float32, shape=(None, img_size_flat), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, n_classes), name='Y')

#    w = tf.get_variable(name='w', shape=(img_size_flat, n_classes), dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
#    b = tf.get_variable(name='b', shape=(n_classes,), dtype=tf.float32, initializer=tf.initializers.zeros())
#    X_w = tf.matmul(X, w, name='X_w')
#    y_logits = tf.add(X_w, b, name='y')

    d1 = tf.layers.dense(X, 512, activation=tf.nn.relu, name='d1', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    y_logits = tf.layers.dense(d1, n_classes, name='d2', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    
    learning_rate = 0.001
    epochs = 1000
    batch_size = 100
    display_freq = 100
    
    # loss function, optimizer and accuracy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=y_logits), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(Y, 1), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # intialize all variables
    init = tf.global_variables_initializer()
    
    #train
    num_tr_iter = int(len(y_train) / batch_size)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            x_train, y_train = randomize(x_train, y_train)

            for iteration in range(num_tr_iter):
                start = iteration * batch_size
                end = start + batch_size
                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
                
                feed_dict_batch = {X: x_batch, Y: y_batch} 
                sess.run(optimizer, feed_dict=feed_dict_batch)

                if iteration % display_freq == 0:
                    loss_batch, acc_batch = sess.run([loss, accuracy], feed_dict=feed_dict_batch)
                    print("Iter {0:3d}:\t Loss={1:.2f}, \tTraining Accuracy={2:.01%}".format(iteration, loss_batch, acc_batch))

            feed_dict_valid = {X: x_valid, Y: y_valid} 
            loss_batch, acc_batch = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
            print("epoch {0:3d}:\t validation Loss={1:.2f}, \tvalidation Accuracy={2:.01%}".format(epoch, loss_batch, acc_batch))
                  
if __name__ == '__main__':
    # Arguments to be parsed via command line
    parser=argparse.ArgumentParser(description="Run Linear Model")
    parser.add_argument("--verbose", "-v", help="verbose output, use -vv and -vvv for more debug information", action="count", default=0)
    parser.set_defaults(func=runModel)

    args = parser.parse_args()
    args.func(args)    
