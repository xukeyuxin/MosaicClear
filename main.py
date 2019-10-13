import tensorflow as tf
import numpy as np
import os
from mosaic import mosaic

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size',16,'default batch_size:16')
tf.flags.DEFINE_integer('epoch',1000,'default epoch:1000')
tf.flags.DEFINE_integer('cpu_nums',3,'default cpu_num:3')
tf.flags.DEFINE_float('lr',1e-3,'default cpu_num:1e-3')
tf.flags.DEFINE_integer('train_data_num',8459,'default train_data_num:8459')
tf.flags.DEFINE_string('model_path','model','default model save path: model')
tf.flags.DEFINE_string('train_utils','gpu','default model save path: /cpu')
tf.flags.DEFINE_string('summary_dir','logs','default model save path: logs')


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        Mosaic = mosaic(sess,FLAGS)
        Mosaic.main()


