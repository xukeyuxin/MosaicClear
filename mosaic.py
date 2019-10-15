import tensorflow as tf
import numpy as np
from op_base import op_base
from utils import *
import layers as ly


class mosaic(op_base):
    def __init__(self, sess, args):
        self.sess = sess
        self.summaries = []
        op_base.__init__(self, args)

    def generator(self, name, input, is_training = True):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            ### 3 -> 64 64
            e0 = ly.conv2d(input, 64, strides=2, name='g_conv2d_0')
            e0 = ly.batch_normal(e0, name='g_bn_0', is_training = is_training)
            e0 = ly.relu(e0, alpha=0.2)

            ### 64 -> 128 32
            e1 = ly.conv2d(e0, 128, strides=2, name='g_conv2d_1')
            e1 = ly.batch_normal(e1, name='g_bn_1', is_training = is_training)
            e1 = ly.relu(e1, alpha=0.2)

            ### 128 -> 256 16
            e2 = ly.conv2d(e1, 256, strides=2, name='g_conv2d_2')
            e2 = ly.batch_normal(e2, name='g_bn_2', is_training = is_training)
            e2 = ly.relu(e2, alpha=0.2)

            ### 256 -> 512 8
            e3 = ly.conv2d(e2, 512, strides=2, name='g_conv2d_3')
            e3 = ly.batch_normal(e3, name='g_bn_3', is_training = is_training)
            e3 = ly.relu(e3, alpha=0.2)

            ### 512 -> 512 4
            e4 = ly.conv2d(e3, 512, strides=2, name='g_conv2d_4')
            e4 = ly.batch_normal(e4, name='g_bn_4', is_training = is_training)
            e4 = ly.relu(e4, alpha=0.2)

            ### 512 -> 512 2
            e5 = ly.conv2d(e4, 512, strides=2, name='g_conv2d_5')
            e5 = ly.batch_normal(e5, name='g_bn_5', is_training = is_training)
            e5 = ly.relu(e5, alpha=0.2)

            ### 512 -> 512 4
            d1 = ly.deconv2d(e5, 512, strides=2, name='g_deconv2d_1')
            d1 = ly.batch_normal(d1, name='g_bn_6', is_training = is_training)
            d1 = tf.nn.dropout(d1, keep_prob=0.5)
            d1 = tf.concat([d1, e4], axis=3)
            d1 = ly.relu(d1, alpha=0.2)

            ### 512 -> 512 8
            d2 = ly.deconv2d(d1, 512, strides=2, name='g_deconv2d_2')
            d2 = ly.batch_normal(d2, name='g_bn_7', is_training = is_training)
            d2 = tf.nn.dropout(d2, keep_prob=0.5)
            d2 = ly.relu(d2, alpha=0.2)
            d2 = tf.concat([d2, e3], axis=3)

            ### 512 -> 256 16
            d3 = ly.deconv2d(d2, 256, strides=2, name='g_deconv2d_3')
            d3 = ly.batch_normal(d3, name='g_bn_8', is_training = is_training)
            d3 = ly.relu(d3, alpha=0.2)
            d3 = tf.concat([d3, e2], axis=3)

            ### 256 -> 128 32
            d4 = ly.deconv2d(d3, 128, strides=2, name='g_deconv2d_4')
            d4 = ly.batch_normal(d4, name='g_bn_9', is_training = is_training)
            d4 = ly.relu(d4, alpha=0.2)
            d4 = tf.concat([d4, e1], axis=3)

            ### 128 -> 64 64
            d5 = ly.deconv2d(d4, 64, strides=2, name='g_deconv2d_5')
            d5 = ly.batch_normal(d5, name='g_bn_10', is_training = is_training)
            d5 = ly.relu(d5, alpha=0.2)
            d5 = tf.concat([d5, e0], axis=3)

            ### 64 -> 3 128
            d6 = ly.deconv2d(d5, 3, strides=2, name='g_deconv2d_6')
            d6 = ly.batch_normal(d6, name='g_bn_11', is_training = is_training)
            d6 = ly.relu(d6, alpha=0.2)

            return tf.nn.tanh(d6)

    def discriminator(self, name, input, target, is_training = True):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            input = tf.concat([input, target], axis=3)
            input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]])
            input = ly.conv2d(input, 64, strides=2, kernal_size=4, padding='VALID', name='d_conv2d_0')
            input = ly.relu(input, alpha=0.2)

            input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]])
            input = ly.conv2d(input, 128, strides=2, name='d_conv2d_1')
            input = ly.batch_normal(input, name='d_bn1',is_training = is_training)
            input = ly.relu(input, alpha=0.2)

            ### 31 | 15
            input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]])
            input = ly.conv2d(input, 256, strides=1, name='d_conv2d_2', padding='VALID')
            input = ly.batch_normal(input, name='d_bn2',is_training = is_training)
            input = ly.relu(input, alpha=0.2)

            ### 30 | 14
            input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]])
            input = ly.conv2d(input, 1, strides=1, name='d_conv2d_3', padding='VALID')

            return tf.nn.sigmoid(input)

    def get_vars(self, name, scope=None):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def discriminator_loss(self, output_data, label_data):
        safe_log = 1e-12
        return - tf.reduce_mean(tf.log(1 - output_data + safe_log)) - tf.reduce_sum(tf.log(label_data + safe_log))

    def generator_loss(self, output_data):
        safe_log = 1e-12
        return - tf.reduce_mean(tf.log(output_data + safe_log))

    def graph(self, input_data, lable_data, d_opt = None, g_opt = None, is_training = True):



        fake = self.generator('G', input_data, is_training = is_training)
        disc_fake = self.discriminator('D', input_data, fake, is_training = is_training)
        disc_real = self.discriminator('D', input_data, lable_data, is_training = is_training)

        d_loss = self.discriminator_loss(disc_fake, disc_real)
        g_loss = self.generator_loss(disc_fake)

        self.summaries.append(tf.summary.scalar('d_loss', d_loss))
        self.summaries.append(tf.summary.scalar('f_loss', g_loss))

        if( not d_opt and not g_opt):
            return

        # d_grads = d_opt.compute_gradients(d_loss)  ## grads, vars
        # g_grads = g_opt.compute_gradients(g_loss)  ## grads, vars
        d_grads = d_opt.compute_gradients(d_loss, var_list=self.get_vars('D'))  ## grads, vars
        g_grads = g_opt.compute_gradients(g_loss, var_list=self.get_vars('G'))  ## grads, vars

        return d_loss, g_loss, d_grads, g_grads

    ### queue
    def build_queue(self, index, batch_size, test = False):

        images = load_image(test)
        weight_cut = images.shape[2] // 2
        input_image = images[:, :, :weight_cut, :], images[:, :, weight_cut:, :]
        mc_image, mc_label = cut_image([index // 2, index % 2], *input_image)

        input_queue = tf.train.slice_input_producer([mc_image, mc_label], num_epochs=self.epoch, shuffle=False)
        image, label = tf.train.batch(input_queue, batch_size=batch_size, num_threads=2,
                                      capacity=64,
                                      allow_smaller_final_batch=False)

        return image, label

    def evaluate(self,image,label):

        self.graph(image, label, is_training = False)

        MOVING_AVERAGE_DECAY = 0.9
        tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY).apply(self.get_vars('G'))
        tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY).apply(self.get_vars('D'))

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        ### init
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
        print('restore success')

        ### 队列启动
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=self.sess)

        ### start test
        try:
            while not coord.should_stop():
                print('start train')
                fake = self.generator('G', image)
                _fake = self.sess.run(fake)
                make_image(_fake, step + '.jpg')

                step += 1


        except tf.errors.OutOfRangeError:
            print('finish test')
        finally:
            coord.request_stop()

        coord.join(thread)





    def train(self, image, label, pretrain=False, need_train = True):


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

        for i in range(self.cpu_nums):
            d_mix_grads = []
            g_mix_grads = []
            with tf.device('%s:%s' % (self.train_utils, i)):
                with tf.name_scope('distributed_%s' % i) as scope:
                    print('start one gpu')
                    # if(need_train):
                    #     need_train = tf.constant(tf.bool,True)
                    # else:
                    #     need_train = tf.constant(tf.bool, False)
                    d_loss, g_loss, d_grads, g_grads = self.graph(image, label, d_opt, g_opt, is_training = need_train)

                    d_mix_grads.append(d_grads)
                    g_mix_grads.append(g_grads)


        d_grads, g_grads = average_gradients(d_mix_grads), average_gradients(g_mix_grads)
        d_grads_op, g_grads_op = d_opt.apply_gradients(d_grads, global_step=global_steps), g_opt.apply_gradients(
            g_grads, global_step=global_steps)

        MOVING_AVERAGE_DECAY = 0.9
        variable_averages_g = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variable_averages_d = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)


        g_variables_averages_op = variable_averages_g.apply(self.get_vars('G'))
        d_variables_averages_op = variable_averages_g.apply(self.get_vars('D'))

        train_op_g = tf.group(g_grads_op, g_variables_averages_op)
        train_op_d = tf.group(d_grads_op, d_variables_averages_op)

        ### summary

        self.summaries.append(tf.summary.scalar('learn_rate', lr))

        for d_grad, d_var in d_grads:
            if d_grad is not None:
                self.summaries.append(tf.summary.histogram(d_var.op.name + '/d_gradients', d_grad))

        for g_grad, g_var in g_grads:
            if g_grad is not None:
                self.summaries.append(tf.summary.histogram(g_var.op.name + '/g_gradients', g_grad))

        for var in tf.trainable_variables():
            self.summaries.append(tf.summary.histogram(var.op.name, var))

        summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        summary_op = tf.summary.merge(self.summaries)

        ### init
        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        self.sess.run([init, init_local])

        ### 队列启动
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=self.sess)

        ### train
        saver = tf.train.Saver(max_to_keep=1)
        if (pretrain):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
            print('restore success %s' % tf.train.latest_checkpoint(self.model_path))

        step = 1

        ### start test
        print('find uninitialized_variable')
        print(self.sess.run(tf.report_uninitialized_variables()))

        if(not need_train):
            try:
                while not coord.should_stop():
                    print('start test')
                    is_training = tf.constant(tf.bool,False)
                    fake = self.generator('G', image, is_training = is_training)
                    _fake = self.sess.run(fake)
                    make_image(_fake, step + '.jpg')

                    step += 1


            except tf.errors.OutOfRangeError:
                print('finish test')
            finally:
                coord.request_stop()

        ### start train
        elif(need_train):
            try:
                while not coord.should_stop():
                    print('start train')
                    _g, _d = self.sess.run([train_op_g, train_op_d])
                    if (step % 10 == 0):
                        print('update summary')
                        summary_str = self.sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                    if (step % 100 == 0):
                        print('update model save')
                        saver.save(self.sess, os.path.join(self.model_path, "model_%s.ckpt" % step))

                    step += 1

            except tf.errors.OutOfRangeError:
                print('finish train')
            finally:
                coord.request_stop()

        coord.join(thread)

    def test(self):
        index = 0
        test_size = len(os.listdir('data/lfw_faces/test'))
        self.batch_size = test_size
        image, label = self.build_queue(index,self.batch_size,test = True)

        self.train(image, label, pretrain=True, need_train = False)

    def main(self):
        index = 0
        image, label = self.build_queue(index,self.batch_size,test = False)
        print(image)
        print(label)
        self.train(image, label, pretrain=False,  need_train = True)
