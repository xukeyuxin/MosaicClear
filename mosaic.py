import tensorflow as tf
import numpy as np
from op_base import op_base
from utils import *


class mosaic(op_base):
    def __init__(self, sess, args):
        self.sess = sess
        op_base.__init__(args)

    def generator(self, name, input):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            return tf.nn.tanh(input)

    def get_vars(self, name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def discriminator(self, name, input):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            return tf.nn.sigmoid(input)

    def discriminator_loss(self, output_data, label_data):
        safe_log = 1e-12
        return - tf.reduce_sum(tf.log(1 - output_data + safe_log)) - tf.reduce_sum(tf.log(label_data + safe_log))

    def generator_loss(self, output_data):
        safe_log = 1e-12
        return - tf.reduce_sum(tf.log(output_data + safe_log))

    def loss(self, scope, input_data, lable_data):
        self.fake = self.generator('G', input_data)
        disc_fake = self.discriminator('D', self.fake)
        disc_real = self.discriminator('D', lable_data)

        d_loss = self.discriminator_loss(disc_fake, disc_real)
        g_loss = self.generator_loss(disc_fake)

        tf.summary.scalar('d_loss', d_loss)
        tf.summary.scalar('f_loss', g_loss)

        return d_loss, g_loss

    ### queue
    def build_queue(self):
        images = []
        label = []
        input_queue = tf.train.slice_input_producer([images, label], num_epochs=self.epoch, shuffle=False)
        input_data, lable_data = tf.train.batch(input_queue, batch_size=self.batch_size, num_threads=1,
                                                capacity=64,
                                                allow_smaller_final_batch=False)
        return input_data, lable_data

    def train(self):

        ## lr
        LEARNING_RATE_DECAY_FACTOR = 0.1
        global_steps = tf.get_variable(name='global_step', shape=[], initializer=tf.constant_initializer(0),
                                       trainable=False)
        decay_change_batch_num = 350.0
        decay_steps = (self.train_data_num / self.batch_size / self.cpu_nums) * decay_change_batch_num

        lr = tf.train.exponential_decay(self.lr,
                                        global_steps,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        ### distributed
        d_opt = tf.train.AdamOptimizer(lr)
        g_opt = tf.train.AdamOptimizer(lr)
        input_queue = self.build_queue()
        for i in range(self.cpu_nums):
            d_mix_grads = []
            g_mix_grads = []
            with tf.device('/%s:%s' % (self.train_utils, i)):
                with tf.variable_scope('%s: %s' % (self.train_utils, i)) as scope:
                    input_data, lable_data = build_queue()
                    d_loss, g_loss = self.loss(input_data, lable_data)
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    summaries.append(tf.summary.scalar('d_loss',d_loss))
                    summaries.append(tf.summary.scalar('g_loss',g_loss))
                    d_grads = d_opt.compute_gradients(d_loss, var_list=self.get_vars('D'))  ## grads, vars
                    g_grads = g_opt.compute_gradients(g_loss, global_step=global_steps,
                                                      var_list=self.get_vars('G'))  ## grads, vars
                    d_mix_grads.append(d_grads)
                    g_mix_grads.append(g_grads)

        d_grads = average_gradients(d_mix_grads)
        g_grads = average_gradients(g_mix_grads)

        d_grads_op = d_opt.apply_gradients(d_grads, global_step=global_steps)
        g_grads_op = g_opt.apply_gradients(g_grads, global_step=global_steps)

        MOVING_AVERAGE_DECAY = 0.9999
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(d_grads_op, g_grads_op, variables_averages_op)

        ### summary
        summaries.append(tf.summary.scalar('learn_rate',lr))

        for d_grad, d_var in d_grads:
            if d_grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/d_gradients', d_grad))

        for g_grad, g_var in g_grads:
            if g_grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/g_gradients', g_grad))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        summary_op = tf.summary.merge(summaries)

        ### init
        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        self.sess.run([init, init_local])

        ### 队列启动
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=self.sess)

        ### train
        saver = tf.Save()
        step = 1
        try:
            while not coord.should_stop():
                _, d_loss, g_loss = sess.run([train_op, d_loss, g_loss])
                step += 1

                if(step % 10 == 0):
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str,step)

                if(step % 100 == 0):
                    saver.save(self.sess,os.path.join(self.model_path,"model_%s.ckpt " % step),global_step = step)

        except tf.errors.OutOfRangeError:
            print('finish train')
        finally:
            coord.request_stop()

        coord.join(thread)
