import tensorflow as tf

class TensorFlowModelTemplate:
    def __init__(self):
        self.init_global_step()
        self.init_cur_epoch()

#    def save(self, sess):
#        print("Saving model...")
#        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
#        print("Model saved")        
#
#    def load(self, sess):
#        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
#        if latest_checkpoint:
#            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
#            self.saver.restor(sess, latest_checkpoint)
#            print("Model loaded")

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.get_variable(initializer=tf.constant(0), trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor+1)
            
    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.get_variable(initializer=tf.constant(0), trainable=False, name='global_step')

#    def init_saver(self):
#        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
