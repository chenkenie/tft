#!/usr/bin/env python

import argparse
import tensorflow as tf
import numpy as np

from trainer_template import SimpleTrainer
from data_loader_template import DataMNIST
from model_template import TensorFlowModelTemplate
from model_template import TensorFlowClassificationModel

img_h = img_w = 28
img_size_flat = img_h * img_w
n_classes = 10
        
class NNModel(TensorFlowClassificationModel):
    def __init__(self):
        super(NNModel, self).__init__()

    def build_model(self):
        # Input
        self.Input = tf.placeholder(tf.float32, shape=(None, img_size_flat), name='INPUT')
        self.Target = tf.placeholder(tf.float32, shape=(None, n_classes), name='TARGET')

        d1 = tf.layers.dense(self.Input, 200, activation=tf.nn.relu, name='d1', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        y_logits = tf.layers.dense(d1, n_classes, name='d2', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        self.set_loss(y_logits)
        
def trainModel(args):

    with tf.Session() as sess:
        data = DataMNIST()
        model = NNModel()
        batch_size = 200

        model.build_model()
        #model.load_weights(sess, 'model_prev')
        trainer = SimpleTrainer(sess, model, data, batch_size, 'models/nn/')

        trainer.train(200)

def validateModel(args):
    with tf.Session() as sess:
        data = DataMNIST()
        model = NNModel()
        model.load(sess, 'models/nn/')
        
        model.predict(sess, data)

    
                  
if __name__ == '__main__':
    # Arguments to be parsed via command line
    parser=argparse.ArgumentParser(description="Run NN Model")
    parser.add_argument("--verbose", "-v", help="verbose output, use -vv and -vvv for more debug information", action="count", default=0)
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_train = subparsers.add_parser('train', help='train model')
    parser_train.set_defaults(func=trainModel)
    parser_validate = subparsers.add_parser('validate', help='validate model')
    parser_validate.set_defaults(func=validateModel)


    args = parser.parse_args()
    args.func(args)    
