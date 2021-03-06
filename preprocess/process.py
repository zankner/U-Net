import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint, random

class Process(object):

  def __init__(self, batchSize, preFetch, imgSize):
    self.batch_size = batchSize
    self.pre_fetch = preFetch
    self.img_size = imgSize
    self.train_pad = 2000

  def build_dataset(self):
    dataset, info = tfds.load('oxford_iiit_pet:3.1.0', with_info=True)
    train_len = info.splits['train'].num_examples
    steps_per_epoch = train_len // self.batch_size
    pure_train = dataset['train']
    train_take = dataset['test'].take(self.train_pad)
    train = pure_train.concatenate(train_take)
    test = dataset['test'].skip(self.train_pad)
    train = train.map(
      self._load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test = test.map(self._load_image_test)
    train = self._prepare(train, True)
    test = self._prepare(test, False)
    return train, test

  def _prepare(self, dataset, train):
    if train:
      dataset = dataset.cache().shuffle(3680 + self.train_pad,reshuffle_each_iteration=True).batch(self.batch_size)
      dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
      dataset = dataset.batch(self.batch_size)
    return dataset

  def _normalize(self, input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

  @tf.function
  def _load_image_train(self, datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
      rotate_factor = randint(0, 4)
      brightness_factor = random() 
      contrast_factor = random() * 3
      saturation_factor = random()
      input_image = tf.image.flip_left_right(input_image)
      input_image = tf.image.flip_up_down(input_image)
      input_image = tf.image.rot90(input_image, rotate_factor)
      input_image = tf.image.adjust_brightness(input_image, brightness_factor)
      input_image = tf.image.adjust_contrast(input_image, contrast_factor)
      input_image = tf.image.adjust_saturation(input_image, saturation_factor)
      input_mask = tf.image.flip_left_right(input_mask)
      input_mask = tf.image.flip_up_down(input_mask)

    input_image, input_mask = self._normalize(input_image, input_mask)

    return input_image, input_mask

  def _load_image_test(self, datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = self._normalize(input_image, input_mask)

    return input_image, input_mask

  def _test(self):
    train, test = self.build_dataset()
    for img, mask in train.take(1):
      plt.figure(figsize=(15, 15))
      title = ['Input Image', 'True Mask']
      plt.subplot(1, 2,1)
      plt.title(title[0])
      plt.imshow(tf.keras.preprocessing.image.array_to_img(img))
      plt.axis('off')
      plt.subplot(1, 2, 2)
      plt.title(title[1])
      plt.imshow(tf.keras.preprocessing.image.array_to_img(mask))
      plt.axis('off')
      plt.show()
