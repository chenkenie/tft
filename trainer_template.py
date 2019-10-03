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
        self.best_loss = 1e32
        self.best_acc = 0.0
        self.best_valid_loss = 1e32
        self.best_valid_acc = 0.0
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

            msg = "Epoch {0:3d}:Training Loss={1:.2f}, Training Accuracy={2:.01%}, Validation Loss={3:.2f}, Validation Accuracy={4:.01%}".format(epoch, epoch_loss, epoch_acc, valid_loss, valid_acc)

            isSaveWeight = False
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                msg += ", Training Loss improved from {:.2f}".format(self.best_loss)
                isSaveWeight = True
            if epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                msg += ", Training Accuracy improved from {:.01%}".format(self.best_acc)
                isSaveWeight = True
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                msg += ", Validation Loss improved from {:.2f}".format(self.best_valid_loss)
                isSaveWeight = True
            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
                msg += ", Validation Accuracy improved from {:.01%}".format(self.best_valid_acc)
                isSaveWeight = True

            if isSaveWeight:
                model.save_weights(sess, self.save_dir, epoch)
                print(msg+", model saved.")
            else:
                print(msg+", Loss or Accuracy was not improve.") 
            
            
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

        loss = np.mean(losses)
        acc = np.mean(accs)

        return loss, acc 

    def train_step(self):
        data = self.data
        model = self.model
        sess = self.sess
        
        x_batch, y_batch = data.next_batch(self.batch_size)

        feed_dict_batch = {model.X: x_batch, model.Y: y_batch} 
        sess.run(model.optimizer, feed_dict=feed_dict_batch)
        loss_batch, acc_batch = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_batch)

        return loss_batch, acc_batch

