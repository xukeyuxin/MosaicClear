import tensorflow as tf
from tqdm import tqdm
import cv2
import os
import numpy as np

def average_gradients(grads):
    average_list = []
    for cell in zip(*grads):
        cell_grads = []
        for grad,var in cell:
            grad = tf.expand_dims(grad,0)
            cell_grads.append(grad)

        average_grad = tf.reduce_mean(tf.concat(cell_grads,axis = 0),axis = 0)
        average_vars = cell[0][1]
        average_list.append((average_grad,average_vars))

    return average_list

def cut_image(cut_index,image,label,cut_num = [2,2]):
    assert image.shape == label.shape,'wrong shape with input_image and label_image'

    image_cut_height = image.shape[1] // cut_num[0]
    image_cut_weight = image.shape[2] // cut_num[1]
    new_image = image[:,cut_index[0] * image_cut_height: (cut_index[0] + 1) * image_cut_height, cut_index[1] * image_cut_weight: (cut_index[1] + 1) * image_cut_weight, :]
    new_label = label[:,cut_index[0] * image_cut_height: (cut_index[0] + 1) * image_cut_height, cut_index[1] * image_cut_weight: (cut_index[1] + 1) * image_cut_weight, :]

    return new_image, new_label

def combine_name(names):
    return '/'.join(names)

def rgb_float(input):
    return (input - 127.5) / 127.5

def float_rgb(input):
    return input * 127.5 + 127.5

def make_image(input):

    print('finish write %s' % index)

def make_image(input):
    image_content = float_rgb(input).astype(np.uint8)
    index = 0
    for cell in image_content:
        index += 1
        cv2.imwrite(os.path.join('data/lfw_build', str(index) + '.jpg'), cell)

def load_image(test):
    image_path = 'data/lfw_faces/train' if not test else 'data/lfw_faces/test'
    image_list = os.listdir(image_path)
    image_content = []
    for i in tqdm(image_list):
        cell_content = cv2.imread(os.path.join(image_path,i)).astype(np.float32)
        image_content.append(cell_content)

    return rgb_float(np.array(image_content))






