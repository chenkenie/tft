import tensorflow as tf
import os

class TensorFlowModelTemplate:
    def __init__(self):
#        self.init_global_step()
#        self.init_cur_epoch()
        pass
    
#    def save(self, sess, checkpoint_dir, epoch):
#        self.saver.save(sess, checkpoint_dir, global_step=epoch, write_meta_graph=True)

    def load(self, sess, checkpoint_dir):
        path = os.path.normpath(checkpoint_dir)
        latest_checkpoint = tf.train.latest_checkpoint(path)
        
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver = tf.train.import_meta_graph(latest_checkpoint+'.meta')
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
        else:
            raise NameError("Loade model weight failed!")

#    def init_cur_epoch(self):
#        with tf.variable_scope('cur_epoch'):
#            self.cur_epoch_tensor = tf.get_variable(initializer=tf.constant(0), trainable=False, name='cur_epoch')
#            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor+1)
#            
#    def init_global_step(self):
#        with tf.variable_scope('global_step'):
#            self.global_step_tensor = tf.get_variable(initializer=tf.constant(0), trainable=False, name='global_step')

#    def init_saver(self):
#        # Saver have to initialted after model is built, otherwise variables are not correctly saved
#        self.saver = tf.train.Saver(max_to_keep=10)

    def build_model(self):
        raise NotImplementedError

    def predict(self, sess, data):
        raise NotImplementedError        



class TensorFlowClassificationModel(TensorFlowModelTemplate):
    def __init__(self):
        super(TensorFlowClassificationModel, self).__init__()

    def set_loss(self, y_logits):

        #require the graph and y_logits to be defined first
        
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
