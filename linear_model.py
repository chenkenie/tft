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
        self.Input = tf.placeholder(tf.float32, shape=(None, img_size_flat), name='INPUT')
        self.Target = tf.placeholder(tf.float32, shape=(None, n_classes), name='TARGET')

        y_logits = tf.layers.dense(self.Input, n_classes, name='d2', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        learning_rate = 0.001
    
        # loss function, optimizer and accuracy
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Target, logits=y_logits), name='loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(self.Target, 1), name='correct_pred')
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    def predict(self, sess, data):
        
        graph = tf.get_default_graph()
        self.Input = graph.get_tensor_by_name("INPUT:0")
        self.Target = graph.get_tensor_by_name("TARGET:0")        
        self.loss = graph.get_tensor_by_name("loss:0")        
        self.accuracy = graph.get_tensor_by_name("accuracy:0")        
        feed_dict_valid = {self.Input: data.input_valid, self.Target: data.target_valid} 
        valid_loss, valid_acc = sess.run([self.loss, self.accuracy], feed_dict=feed_dict_valid)

        msg = "Validation Points {}, Loss={:.2f}, Accuracy={:.01%}".format(data.input_valid.shape[0], valid_loss, valid_acc)
        print(msg)
        
        
def trainModel(args):

    with tf.Session() as sess:
        data = DataMNIST()
        model = LinearModel()
        batch_size = 200

        model.build_model()
        trainer = SimpleTrainer(sess, model, data, batch_size, 'models/linear/')

        trainer.train(200)

def validateModel(args):
    with tf.Session() as sess:
        data = DataMNIST()
        model = LinearModel()
        model.load(sess, 'models/linear/')
        
        model.predict(sess, data)
        
                  
if __name__ == '__main__':
    # Arguments to be parsed via command line
    parser=argparse.ArgumentParser(description="Run Linear Model")
    parser.add_argument("--verbose", "-v", help="verbose output, use -vv and -vvv for more debug information", action="count", default=0)
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_train = subparsers.add_parser('train', help='train model')
    parser_train.set_defaults(func=trainModel)
    parser_validate = subparsers.add_parser('validate', help='validate model')
    parser_validate.set_defaults(func=validateModel)


    args = parser.parse_args()
    args.func(args)    
