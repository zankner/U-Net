import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Conv2DTranspose, 
    BatchNormalization, concatenate, MaxPool2D, Dropout
)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1_l2


class UNet(Model):
  def __init__(self, filters=16):
    super(UNet, self).__init__()
    self.run_eagerly = False
    #Downsampling step
    self.encoder = []
    for i in range(4):
      ds_step = []
      ds_step.append(Conv2D((2**i)*filters, 3, padding='same', kernel_regularizer=l1_l2(0.05, 0.05)))
      ds_step.append(BatchNormalization())
      ds_step.append(Conv2D((2**i)*filters, 3, padding='same', kernel_regularizer=l1_l2(0.05, 0.05)))
      ds_step.append(BatchNormalization())
      ds_step.append(MaxPool2D())
      self.encoder.append(ds_step)
    self.encoder_bot_0 = Conv2D(filters*16, 3, padding='same', kernel_regularizer=l1_l2(0.05, 0.05))
    self.encoder_bot_1 = Conv2D(filters*16, 3, padding='same', kernel_regularizer=l1_l2(0.05, 0.05))
    self.encoder_bot_batchnorm = BatchNormalization()
   
    #Upsampling step
    self.decoder = []
    for i in range(4):
      up_step = []
      up_step.append(Conv2DTranspose((2**(3-i))*16, 3, 2, padding='same', kernel_regularizer=l1_l2(0.05, 0.05)))
      up_step.append(Conv2D((2**(3-i))*16, 3, padding='same', kernel_regularizer=l1_l2(0.05, 0.05)))
      up_step.append(BatchNormalization())
      up_step.append(Conv2D((2**(3-i))*16, 3, padding='same', kernel_regularizer=l1_l2(0.05, 0.05)))
      up_step.append(BatchNormalization())
      self.decoder.append(up_step)
    self.out_conv = Conv2D(3, 1, activation='softmax')


  def call(self, x, training=False):
    skip_connections = []

    #Downsampling
    for ds_step in self.encoder:
      x = ds_step[0](x)
      if training:
        x = ds_step[1](x)
        Dropout(.7)(x)
      x = tf.nn.relu(x)
      x = ds_step[2](x)
      if training:
        x = ds_step[3](x)
        Dropout(.7)(x)
      x = tf.nn.relu(x)
      skip_connections.append(x)
      x = ds_step[4](x)
      x = tf.nn.dropout(x, 0.7)
    x = self.encoder_bot_0(x)
    x = self.encoder_bot_1(x)
    if training:
        x = self.encoder_bot_batchnorm(x)

    #Upsampling
    skip_connections = reversed(skip_connections)
    for up_step, skip in zip(self.decoder, skip_connections):
      x = up_step[0](x)
      x = concatenate([x, skip])
      if training:
        x = tf.nn.dropout(x, .7)
      x = up_step[1](x)
      if training:
        x = up_step[2](x)
        Dropout(.7)(x)
      x = tf.nn.relu(x)
      x = up_step[3](x)
      if training:
        x = up_step[4](x)
        Dropout(.7)(x)
      x = tf.nn.relu(x)
    x = self.out_conv(x)

    return x
