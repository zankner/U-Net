import os
import argparse
import logging
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import (
  Mean, SparseCategoricalAccuracy
)
from models.UNet import UNet
from preprocess.process import Process

class Train(object):
    def __init__(self, params):
        self.lr = params.lr
        self.epochs = params.epochs
        self.loss_object = SparseCategoricalCrossentropy()
        self.optimizer = Adam()
        self.train_loss = Mean(name='train_loss')
        self.train_accuracy = SparseCategoricalAccuracy(name='train_acc')
        self.test_loss = Mean(name='test_loss')
        self.test_accuracy = SparseCategoricalAccuracy(name='test_acc')
        self.model = UNet()
        preprocessor = Process(params.batchSize, params.preFetch, 128)
        self.train_ds, self.test_ds = preprocessor.build_dataset()
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer,
                net=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, f'checkpoints{params.ckpt_dir}',
                max_to_keep=3)

    @tf.function
    def _update(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def _test(self, inputs, labels):
        predictions = self.model(labels)
        loss = self.loss_object(labels, predictions)

        self.test_loss(loss)
        self.test_accuracy(labels, predictions)

    def _log(self, epoch):
        template = 'Epoch {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'
        print(template.format(epoch + 1,
            self.train_loss.result(),
            self.train_accuracy.result() * 100,
            self.test_loss.result(),
            self.test_accuracy.result() * 100))

    def _save(self):
        if int(self.ckpt.step) % 10 == 0:
            save_path = self.ckpt_manager().save()
            ckptLog = f"Saved checkpoint for step {int(self.ckpt.step)}: {save_path}"
            print(ckptLog)

    def _restore(self):
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print(f"Restored from {manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

    def _reset(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

    def train(self):
        self._restore()
        for epoch in range(self.epochs):
            for inputs, labels in self.train_ds:
                self._update(inputs, labels)
            for testInputs, testLabels in self.test_ds:
                self._test(testInput, testLabels)
            self._log(epoch)
            self._save()
            self._reset()


if __name__ == '__main__':
    numCkpts = len([folder for folder in os.listdir('./checkpoints') if os.path.isdir(folder)])
    parser= argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--ckpt_dir', default=str(numCkpts), type=str)
    parser.add_argument('--dataDir', default='./data/train', type=str)
    parser.add_argument('--batchSize', default=32, type=int)
    parser.add_argument('--preFetch', default=1, type=int)
    parser.add_argument('--imgSize', default=128, type=int)
    parser.add_argument('--testRatio', default=0.1, type=float)
    parser.add_argument('--maxDelta', default=32.0 / 255.0, type=float)
    parser.add_argument('--lowerSat', default=0.5, type=float)
    parser.add_argument('--upperSat', default=1.5, type=float)
    args = parser.parse_args()
    actuator = Train(args)
    actuator.train()
