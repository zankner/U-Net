import os
import argparse
from IPython.display import clear_output
import tensorflow as tf
from tensorflow.keras.metrics import (
  SparseCategoricalAccuracy
)
from models.UNet import UNet
from preprocess.process import Process
import matplotlib.pyplot as plt


class Test(object):
    def __init__(self):
      preprocessor = Process(1, 0, 128)
      self.test_accuracy = SparseCategoricalAccuracy(name='test_acc')
      self.train_ds, self.test_ds = preprocessor.build_dataset()
      self.model = UNet()
      self.ckpt = tf.train.Checkpoint(
        net=self.model)

    def _test(self, input_, label):
      clear_output()
      prediction = self.model(input_)
      self.test_accuracy(label, prediction)
      prediction = self._convert_prediction_mask(prediction)
      input_ = tf.squeeze(input_)
      label = label[0]
      #self._display([input_, label, prediction])

    def _display(self, display_list):
      plt.figure(figsize=(15, 15))
      title = ['Input Image', 'True Mask', 'Predicted Mask']
      for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
      plt.show()

    def _convert_prediction_mask(self, pred_mask):
      pred_mask = tf.argmax(pred_mask, axis=-1)
      pred_mask = pred_mask[..., tf.newaxis]
      return pred_mask[0]

    def _log(self):
     print('*********************') 
     print(f'Acc: {self.test_accuracy.result() * 100})')

    def _restore(self):
      self.ckpt.restore('final-weights/57-val-acc/ckpt-1000').expect_partial()
      #self.model.compile(loss='sparse_categorical_crossentropy',
      #  optimizer=tf.keras.optimizers.Adam(1e-4))
      #self.model.predict(tf.random.uniform(
      #  [1, 128, 128, 3]
      #))
      #self.model.load_weights('final-weights/57-val-acc/ckpt-1000')

    def test(self):
      self._restore()
      print('**************8')
      print(len(list(self.test_ds)))
      for testInputs, testLabels in self.test_ds:
        print('**************8')
        self._test(testInputs, testLabels)
      self._log()


if __name__ == '__main__':
    actuator = Test()
    actuator.test()
