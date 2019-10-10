import tensorflow as tf
import numpy as np
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size',16)
tf.flags.DEFINE_string('model_path','model')
