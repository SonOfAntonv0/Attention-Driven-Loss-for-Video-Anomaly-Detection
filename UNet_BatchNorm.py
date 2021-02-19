import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.layers import Conv2DTranspose,BatchNormalization,Activation
class UNet():
    def __init__(self,img_shape,num_c=1):
        print (f'Input shape is {img_shape[1:]}')
        print(f'no of classes is {num_c}')
        self.image_shape=img_shape[1:]
        self.num_c=num_c

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        print(f'cw is {target.get_shape()[2] - refer.get_shape()[2]}')
        cw = (target.get_shape()[2] - refer.get_shape()[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = cw//2, cw//2 + 1
        else:
            cw1, cw2 = cw//2, cw//2
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = ch//2, ch//2 + 1
        else:
            ch1, ch2 = ch//2, ch//2

        return (ch1, ch2), (cw1, cw2)
#Relu is being replaced with None for EvoNorm2dS0
    def create_model(self):
        img_shape=self.image_shape
        num_class=self.num_c
        concat_axis = 3
        inputs = layers.Input(shape = img_shape)

        conv1 = layers.Conv2D(64, (3, 3), activation=None, padding='same', name='conv1_1')(inputs)
        print(conv1.shape)
        conv1=BatchNormalization(axis=-1)(conv1)
        conv1=Activation(tf.nn.relu)(conv1)
        conv1 = layers.Conv2D(32, (3, 3), activation=None, padding='same')(conv1)
        conv1=BatchNormalization(axis=-1)(conv1)
        conv1=Activation(tf.nn.relu)(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = layers.Conv2D(64, (3, 3), activation=None, padding='same')(pool1)
        conv2=BatchNormalization(axis=-1)(conv2)
        conv2=Activation(tf.nn.relu)(conv2)
        conv2 = layers.Conv2D(64, (3, 3), activation=None, padding='same')(conv2)
        conv2=BatchNormalization(axis=1)(conv2)
        conv2=Activation(tf.nn.relu)(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(128, (3, 3), activation=None, padding='same')(pool2)
        conv3=BatchNormalization(axis=-1)(conv3)
        conv3=Activation(tf.nn.relu)(conv3)
        conv3 = layers.Conv2D(128, (3, 3), activation=None, padding='same')(conv3)
        conv3=BatchNormalization(axis=-1)(conv3)
        conv3=Activation(tf.nn.relu)(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(256, (3, 3), activation=None, padding='same')(pool3)
        conv4=BatchNormalization(axis=-1)(conv4)
        conv4=Activation(tf.nn.relu)(conv4)
        conv4 = layers.Conv2D(256, (3, 3), activation=None, padding='same')(conv4)
        conv4=BatchNormalization(axis=-1)(conv4)
        conv4=Activation(tf.nn.relu)(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(512, (3, 3), activation=None, padding='same')(pool4)
        conv5=BatchNormalization(axis=-1)(conv5)
        conv5=Activation(tf.nn.relu)(conv5)
        conv5 = layers.Conv2D(512, (3, 3), activation=None, padding='same')(conv5)
        conv5=BatchNormalization(axis=-1)(conv5)
        conv5=Activation(tf.nn.relu)(conv5)
        up_conv5 = layers.UpSampling2D(size=(2, 2))(conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = layers.Cropping2D(cropping=(ch,cw))(conv4)
        up6 = layers.concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = layers.Conv2D(256, (3, 3), activation=None, padding='same')(up6)
        conv6=BatchNormalization(axis=-1)(conv6)
        conv6=Activation(tf.nn.relu)(conv6)
        conv6 = layers.Conv2D(256, (3, 3), activation=None, padding='same')(conv6)
        conv6=BatchNormalization(axis=-1)(conv6)
        conv6=Activation(tf.nn.relu)(conv6)
        up_conv6 = layers.UpSampling2D(size=(2, 2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = layers.Cropping2D(cropping=(ch,cw))(conv3)
        up7 = layers.concatenate([up_conv6, crop_conv3], axis=concat_axis) 
        conv7 = layers.Conv2D(128, (3, 3), activation=None, padding='same')(up7)
        conv7=BatchNormalization(axis=-1)(conv7)
        conv7=Activation(tf.nn.relu)(conv7)
        conv7 = layers.Conv2D(128, (3, 3), activation=None, padding='same')(conv7)
        conv7=BatchNormalization(axis=-1)(conv7)
        conv7=Activation(tf.nn.relu)(conv7)
        up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = layers.Cropping2D(cropping=(ch,cw))(conv2)
        up8 = layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = layers.Conv2D(64, (3, 3), activation=None, padding='same')(up8)
        conv8=BatchNormalization(axis=-1)(conv8)
        conv8=Activation(tf.nn.relu)(conv8)
        conv8 = layers.Conv2D(64, (3, 3), activation=None, padding='same')(conv8)
        conv8=BatchNormalization(axis=-1)(conv8)
        conv8=Activation(tf.nn.relu)(conv8)
        up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = layers.Cropping2D(cropping=(ch,cw))(conv1)
        up9 = layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = layers.Conv2D(32, (3, 3), activation=None, padding='same')(up9)
        conv9=BatchNormalization(axis=-1)(conv9)
        conv9=Activation(tf.nn.relu)(conv9)
        conv9 = layers.Conv2D(32, (3, 3), activation=None, padding='same')(conv9)
        conv9=BatchNormalization(axis=-1)(conv9)
        conv9=Activation(tf.nn.relu)(conv9)
        ch, cw = self.get_crop_shape(inputs, conv9)
        conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = layers.Conv2D(num_class, (1, 1))(conv9)
        model = models.Model(inputs=inputs, outputs=conv10)

        return model

