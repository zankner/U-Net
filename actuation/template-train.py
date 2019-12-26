import argparse
import tensorflow as tf
from tf.keras.losses import ()
from tf.keras.optimizers import ()
from tf.train import (Checkpoint, CheckpointManager)
from models. import
from preprocess.process import Process


class train(object):
    def __init__(self, params):
        self.lr = params['lr']
        self.lossObject =
        self.optimizer =
        self.trainLoss =
        self.trainAccuracy =
        self.testLoss =
        self.testAccuracy =
        self.model = TemplateModel()
        dataDir = [params['dataDir'] + '/inputs',
            params['dataDir'] + '/train.csv']
        preprocessor = Process(dataDir, params['batchSize'],
            params['preFetch'], params['imgSize'], params['testRatio'],
            params['maxDelta'], params['lowerSat'], params['upperSat'])
        self.trainDs, self.testDs = preprocessor.buildDataset()
        self.ckpt = Checkpoint(step=tf.Variable(1), optimizer=self.optimizer,
                net=self.model)
        self.ckptManager = CheckpointManager(ckpt, f'checkpoints{params["ckptDir"]}',
                max_to_keep=3)

    @tf.function
    def _update(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = self.lossObject(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.trainLoss(loss)
        self.trainAccuracy(labels, predictions)

    @tf.function
    def _test(self, inputs, labels):
        predictions = self.model(labels)
        loss = self.loss_object(labels, predictions)

        self.testLoss(loss)
        self.testAccuracy(labels, predictions)

    def _log(self, epoch):
        template = 'Epoch {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'
        print(template.format(epoch + 1,
            self.trainLoss.result(),
            self.trainAccuracy.result() * 100,
            self.testLoss.result(),
            self.testAccuracy.result() * 100))

    def _save(self):
        if int(self.ckpt.step) % 10 == 0:
            save_path = self.ckptManager().save()
            ckptLog = f"Saved checkpoint for step {int(self.ckpt.step)}: {save_path}"
            print(ckptLog)

    def _restore(self):
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print(f"Restored from {manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

    def _reset(self):
        self.trainLoss.reset_states()
        self.trainAccuracy.reset_states()
        self.testLoss.reset_states()
        self.testAccuracy.reset_states()

    def train(self):
        self._restore()
        for epoch in range(self.epochs):
            for inputs, labels in self.trainDs:
                _update(inputs, labels)
            for testInputs, testLabels in self.testDs:
                _test(testInput, testLabels)
            self._log(epoch)
            self._save()
            self._reset()


if __name__ == '__main__':
    numCkpts = len([folder for folder in os.listdir('./checkpoints')
        if os.path.isdir(folder))
    parser= argparse.ArumentParser()
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--ckptDir', default=str(numCkpts), type=str)
    parser.add_argument('--dataDir', default='./data/train', type=str)
    parser.add_argument('--batchSize', default=32, type=int)
    parser.add_argument('--preFetch', default=1, type=int)
    parser.add_argument('--imgSize', default=128, type=int)
    parser.add_argument('--testRatio', default=0.1, type=float)
    parser.add_argument('--maxDelta', default=32.0 / 255.0, type=float)
    parser.add_argument('--lowerSat', default=0.5, type=float)
    parser.add_argument('--upperSat', default=1.5, type=float)
