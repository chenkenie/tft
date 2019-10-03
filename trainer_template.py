import tensorflow as tf
import numpy as np
from tqdm import tqdm, trange


class TensorFlowTrainerTemplate:
    def __init__(self, sess, model, data, batch_size, save_dir):
        self.model = model
        self.sess = sess
        self.data = data
        self.batch_size = batch_size
        self.save_dir = save_dir
        
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self, epochs):
        for epoch in range(epochs):
            data = self.data
            model = self.model
            sess = self.sess

            data.epoch_init()
            epoch_loss, epoch_acc = self.train_epoch(epoch)
            
            feed_dict_valid = {model.X: data.x_valid, model.Y: data.y_valid} 
            valid_loss, valid_acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_valid)
            print("Epoch {0:3d}:\tTraining Loss={1:.2f}, \tTraining Accuracy={2:.01%} \tValidation Loss={3:.2f}, \tValidation Accuracy={4:.01%}".format(epoch, epoch_loss, epoch_acc, valid_loss, valid_acc))
            
    def train_epoch(self, epoch):
        data = self.data
        model = self.model
        sess = self.sess

        losses = []
        accs = []

        batchCnt = int(len(data.y_train) / self.batch_size)
        
        with trange(batchCnt, ncols=100, ascii=True) as t:
            t.set_description("Epoch {:3d}".format(epoch))
                
            for iteration in t:
                loss_batch, acc_batch = self.train_step()

                losses.append(loss_batch)
                accs.append(acc_batch)

        model.save_weights(sess, self.save_dir, epoch)
        
        return np.mean(losses), np.mean(accs)

    def train_step(self):
        data = self.data
        model = self.model
        sess = self.sess
        
        x_batch, y_batch = data.next_batch(self.batch_size)

        feed_dict_batch = {model.X: x_batch, model.Y: y_batch} 
        sess.run(model.optimizer, feed_dict=feed_dict_batch)
        loss_batch, acc_batch = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_batch)

        return loss_batch, acc_batch

