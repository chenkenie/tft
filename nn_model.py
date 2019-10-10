#!/usr/bin/env python

import argparse
import tensorflow as tf
import numpy as np

from trainer_template import SimpleTrainer
from data_loader_template import DataMNIST
from model_template import TensorFlowModelTemplate

img_h = img_w = 28
img_size_flat = img_h * img_w
n_classes = 10
        
class LinearModel(TensorFlowModelTemplate):
    def __init__(self):
        super(LinearModel, self).__init__()

    def build_model(self):
        # Input
        self.X = tf.placeholder(tf.float32, shape=(None, img_size_flat), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None, n_classes), name='Y')

        d1 = tf.layers.dense(self.X, 200, activation=tf.nn.relu, name='d1', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        y_logits = tf.layers.dense(d1, n_classes, name='d2', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        learning_rate = 0.001
    
        # loss function, optimizer and accuracy
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=y_logits), name='loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(self.Y, 1), name='correct_pred')
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        
def runModel(args):

    with tf.Session() as sess:
        data = DataMNIST()
        model = LinearModel()
        batch_size = 200

        model.build_model()
        #model.load_weights(sess, 'model_prev')
        trainer = SimpleTrainer(sess, model, data, batch_size, 'models/nn')

        trainer.train(200)
        
                  
if __name__ == '__main__':
    # Arguments to be parsed via command line
    parser=argparse.ArgumentParser(description="Run Linear Model")
    parser.add_argument("--verbose", "-v", help="verbose output, use -vv and -vvv for more debug information", action="count", default=0)
    parser.set_defaults(func=runModel)

    args = parser.parse_args()
    args.func(args)    
