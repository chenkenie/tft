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
    

from model_template import TensorFlowModelTemplate

class LinearModel(TensorFlowModelTemplate):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.build_model()
#        self.init_saver()

#    def init_saver(self):
#        self.saver = tr.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        # Input
        self.X = tf.placeholder(tf.float32, shape=(None, img_size_flat), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None, n_classes), name='Y')

        d1 = tf.layers.dense(self.X, 512, activation=tf.nn.relu, name='d1', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        y_logits = tf.layers.dense(d1, n_classes, name='d2', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        learning_rate = 0.001
    
        # loss function, optimizer and accuracy
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=y_logits), name='loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(self.Y, 1), name='correct_pred')
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        

class DataGenerator:
    def __init__(self):
        self.x_train, self.y_train, self.x_valid, self.y_valid = load_data()

        self.x_train = self.x_train.reshape((-1, img_size_flat))
        self.x_valid = self.x_valid.reshape((-1, img_size_flat))

    def epoch_init(self):
        self.x_train, self.y_train = randomize(self.x_train, self.y_train)
        self.start = 0
        self.end = 0
        
    def next_batch(self, batch_size):
        self.start = self.end
        self.end = self.start + batch_size
        x_batch, y_batch = get_next_batch(self.x_train, self.y_train, self.start, self.end)
        return x_batch, y_batch

class TensorFlowTrainerTemplate:
    def __init__(self, sess, model, data):
        self.model = model
        self.sess = sess
        self.data = data
        self.batch_size = 100
        self.num_tr_iter = int(len(self.data.y_train) / self.batch_size)
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        epochs = 1000
        for epoch in range(epochs):
            self.data.epoch_init()
            self.train_epoch()

            data = self.data
            model = self.model
            sess = self.sess
            
            feed_dict_valid = {model.X: data.x_valid, model.Y: data.y_valid} 
            loss_batch, acc_batch = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_valid)
            print("epoch {0:3d}:\t validation Loss={1:.2f}, \tvalidation Accuracy={2:.01%}".format(epoch, loss_batch, acc_batch))
            
    def train_epoch(self):
        display_freq = 100

        for iteration in range(self.num_tr_iter):
            self.train_step()

    

    def train_step(self):
        x_batch, y_batch = self.data.next_batch(self.batch_size)

        data = self.data
        model = self.model
        sess = self.sess
        
        feed_dict_batch = {model.X: x_batch, model.Y: y_batch} 
        sess.run(model.optimizer, feed_dict=feed_dict_batch)

        #calc iter loss
        #calc epoch average loss
        
#        if iteration % display_freq == 0:
#            loss_batch, acc_batch = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_batch)
#            print("Iter {0:3d}:\t Loss={1:.2f}, \tTraining Accuracy={2:.01%}".format(iteration, loss_batch, acc_batch))
    
        
def runModel(args):

    with tf.Session() as sess:
        data = DataGenerator()
        model = LinearModel()
        trainer = TensorFlowTrainerTemplate(sess, model, data)

        trainer.train()
        
#    x_train, y_train, x_valid, y_valid = load_data()
#
#    x_train = x_train.reshape((-1, img_size_flat))
#    x_valid = x_valid.reshape((-1, img_size_flat))
#    
#    print("Shape of Input:")
#    print("- Training-set:\t\t{}, {}".format(x_train.shape, y_train.shape))
#    print("- Validation-set:\t{}, {}".format(x_valid.shape, y_valid.shape))    
#
#    model = LinearModel()
#    
#    # intialize all variables
#    init = tf.global_variables_initializer()
#    
#    epochs = 1000
#    batch_size = 100
#    display_freq = 100
#    
#     #train
#    num_tr_iter = int(len(y_train) / batch_size)
#    with tf.Session() as sess:
#        sess.run(init)
#        for epoch in range(epochs):
#            x_train, y_train = randomize(x_train, y_train)
#
#            for iteration in range(num_tr_iter):
#                start = iteration * batch_size
#                end = start + batch_size
#                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
#                
#                feed_dict_batch = {model.X: x_batch, model.Y: y_batch} 
#                sess.run(model.optimizer, feed_dict=feed_dict_batch)
#
#                if iteration % display_freq == 0:
#                    loss_batch, acc_batch = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_batch)
#                    print("Iter {0:3d}:\t Loss={1:.2f}, \tTraining Accuracy={2:.01%}".format(iteration, loss_batch, acc_batch))
#
#            feed_dict_valid = {model.X: x_valid, model.Y: y_valid} 
#            loss_batch, acc_batch = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_valid)
#            print("epoch {0:3d}:\t validation Loss={1:.2f}, \tvalidation Accuracy={2:.01%}".format(epoch, loss_batch, acc_batch))
                  
if __name__ == '__main__':
    # Arguments to be parsed via command line
    parser=argparse.ArgumentParser(description="Run Linear Model")
    parser.add_argument("--verbose", "-v", help="verbose output, use -vv and -vvv for more debug information", action="count", default=0)
    parser.set_defaults(func=runModel)

    args = parser.parse_args()
    args.func(args)    
