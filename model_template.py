import tensorflow as tf

class TensorFlowModelTemplate:
    def __init__(self):
        self.init_global_step()
        self.init_cur_epoch()
        self.init_saver()

    def save_weights(self, sess, checkpoint_dir, epoch):
        self.saver.save(sess, checkpoint_dir, global_step=epoch, write_meta_graph=False)

    def load_weights(self, sess, checkpoint_dir):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.get_variable(initializer=tf.constant(0), trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor+1)
            
    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.get_variable(initializer=tf.constant(0), trainable=False, name='global_step')

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=10)

    def build_model(self):
        raise NotImplementedError
