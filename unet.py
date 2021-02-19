import tensorflow as tf
from tensorflow.keras.layers import Conv2D as conv2d 
from tensorflow.keras.layers import MaxPool2D as max_pool2d
from tensorflow.keras.layers import Conv2DTranspose as conv2d_transpose

def unet(inputs, layers, features_root=64, filter_size=3, pool_size=2, output_channel=1):
    """
    :param inputs: input tensor, shape[None, height, width, channel]
    :param layers: number of layers
    :param features_root: number of features in the first layer
    :param filter_size: size of each conv layer
    :param pool_size:  size of each max pooling layer
    :param output_channel:  number of channel for output tensor
    :return: a tensor, shape[None, height, width, output_channel]
    """

    in_node = inputs
    conv = []
    for layer in range(0, layers):
        features = 2**layer*features_root

        conv1 = conv2d(filters=features, kernel_size=filter_size)(in_node)#conv2d(inputs=in_node, num_outputs=features, kernel_size=filter_size)
        print(f'Conv1 shape is {conv1.shape}')
        conv2 = conv2d(filters=features, kernel_size=filter_size)(conv1)#conv2d(inputs=conv1, num_outputs=features, kernel_size=filter_size)
        conv.append(conv2)
        print(f'Conv2 shape is {conv2.shape}')

        if layer < layers - 1:
            in_node = max_pool2d(pool_size=pool_size, padding='SAME')(conv2)#max_pool2d(inputs=conv2, kernel_size=pool_size, padding='SAME')
            # in_node = conv2d(inputs=conv2, num_outputs=features, kernel_size=filter_size, stride=2)
        print(f'in_node shape is {in_node.shape}')
    in_node = conv[-1]

    for layer in range(layers-2, -1, -1):
        features = 2**(layer+1)*features_root

        h_deconv = conv2d_transpose(filters=features//2, kernel_size=pool_size, strides=pool_size)(in_node)
        print(f'h_deconv shape is {h_deconv.shape}')#conv2d_transpose(inputs=in_node, num_outputs=features//2, kernel_size=pool_size, stride=pool_size)
        h_deconv_concat = tf.concat([conv[layer], h_deconv], axis=3)

        conv1 = conv2d(filters=features//2, kernel_size=filter_size)(h_deconv_concat)#conv2d(inputs=h_deconv_concat, num_outputs=features//2, kernel_size=filter_size)
        in_node = conv2d(inputs=conv1, num_outputs=features//2, kernel_size=filter_size)(conv1)#conv2d(inputs=conv1, num_outputs=features//2, kernel_size=filter_size)

    output = conv2d(inputs=in_node, num_outputs=output_channel, kernel_size=filter_size, activation_fn=None)(in_node)#conv2d(inputs=in_node, num_outputs=output_channel, kernel_size=filter_size, activation_fn=None)
    output = tf.tanh(output)
    return output
