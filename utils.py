import tensorflow as tf

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


