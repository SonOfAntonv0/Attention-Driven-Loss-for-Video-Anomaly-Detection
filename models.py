import tensorflow as tf

from UNet_nonorm import UNet
import pix2pix_nonorm

from flownet2.src.flowlib import flow_to_image
from flownet2.src.flownet_sd.flownet_sd import FlowNetSD  # Ok
from flownet2.src.training_schedules import LONG_SCHEDULE
from flownet2.src.net import Mode

import tf_slim as slim
print("Hello")

def generator(inputs, layers, features_root=64, filter_size=3, pool_size=2, output_channel=3):
    m=UNet(inputs.shape,output_channel)
    return m.create_model()(inputs)


def discriminator(inputs, num_filers=(128, 256, 512, 512)):
    logits, end_points = pix2pix_nonorm.pix2pix_discriminator(inputs, num_filers)
    return logits, end_points['predictions']


def flownet(input_a, input_b, height, width, reuse=None):
    net = FlowNetSD(mode=Mode.TEST)
    # train preds flow
    input_a = (input_a + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    input_b = (input_b + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    # input size is 384 x 512
    input_a = tf.compat.v1.image.resize_images(input_a, [height, width])
    input_b = tf.compat.v1.image.resize_images(input_b, [height, width])
    flows = net.model(
        inputs={'input_a': input_a, 'input_b': input_b},
        training_schedule=LONG_SCHEDULE,
        trainable=False, reuse=reuse
    )
    return flows['flow']


def initialize_flownet(sess, checkpoint):
    flownet_vars = slim.get_variables_to_restore(include=['FlowNetSD'])
    flownet_saver = tf.compat.v1.train.Saver(flownet_vars)
    print('FlownetSD restore from {}!'.format(checkpoint))
    flownet_saver.restore(sess, checkpoint)
