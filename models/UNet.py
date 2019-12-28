import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D
)
from tensorflow.keras import Model


class UNet(Model):
  def __init__(self, train):
    #Downsampling step
    self.encoder = []
    for i in range(4):
      ds_step = []
      ds_step.append(Conv2D(2**i, 3, padding='same'))
      ds_step.append(Conv2D(2**i, 3, padding='same'))
      self.encoder.append(ds_step)
    self.encoder_bot_0 = Conv2D(filters*16, 3))
    self.encoder_bot_1 = Conv2D(filters*16, 3))
    
    #Upsampling step
    self.decoder = []
    for i in range(4):
      up_step = []
      up_step.append(Conv2DTranspose(2**(3-i), 3, 2, padding='same'))
      us_step.append(Conv2D(2**(3-i), 3, padding='same'))
      us_step.append(Conv2D(2**(3-i), 3, padding='same'))
      self.decoder.append(up_step)
    self.out_conv = Conv2D(1, 1, activation='softmax')

  def call(self, x, training=False):
    skip_connections = []

    #Downsampling
    for ds_step in self.encoder:
      x = ds_step[0]
      if training:
        x = BatchNormalization(x)
      x = relu(x)
      x = ds_step[1]
      if training:
        x = BatchNormalization(x)
      x = relu(x)
      skip_connections.append(x)
    x = self.encoder_bot_0(x)
    x = self.encoder_bot_1(x)

    #Upsampling
    skip_connections = reversed(skip_connections)
    for up_step, skip in self.decoder, skip_connections:
      x = up_step[0]
      x = tf.concat([x, skip])
      if training:
        x = dropout(x)
      x = up_step[1]
      if training:
        x = BatchNormalization(x)
      x = relu(x)
      x = up_step[2]
      if training:
        x = BatchNormalization(x)
      x = relu(x)

    x = self.out_conv(x)

    return x
