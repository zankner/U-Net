from tf.data import Dataset
from tf.image import (
    decode_jpeg,
    convert_image_dtype,
    resize_images,
    random_brightness,
    random_saturation
)
import random
import os
import csv


class Process(object):

    def __init__(self, dataDir, maskDir, batchSize, preFetch, imgSize,
                 testRatio, maxDelta, lowerSat, upperSat):
        self.dataDir = dataDir
        self.batchSize = batchSize
        self.preFetch = preFetch
        self.imgSize = imgSize
        self.testRatio = testRatio
        self.maxDelta = maxDelta
        self.lower = lowerSat
        self.upper = upperSat
        self.maskFiles = [filename for filename in os.listdir(maskDir) if os.isfile(filename)]

    def buildDataset(self):
        inputs = []
        outputs = []
        with open(self.dataDir[1], 'r') as f:
            csvReader = csv.reader(f, delimiter=',')
            for row in csvReader:
                inputs.append(self.dataDir[0] + row[0])
                outputs.append(row[1])
        data = zip(inputs, outputs)
        random.shuffle(data)
        inputs, outputs = zip(*data)
        dataset = Dataset.from_tensor_slices((inputs, outputs))
        dataset = dataset.map(self._parseData, num_parallel_calls=4)
        dataset = dataset.map(self._preproces, num_parallel_calls=4)
        dataset = dataset.batch(self.batchSize)
        dataset = dataset.prefetch(self.preFetch)
        testLen = int(len(inputs) * self.testRatio)
        trainLen = len(inputs) - testLen
        testDataset = dataset.take(testLen)
        testDataset = testDataset.shuffle(testLen)
        trainDataset = dataset.skip(testLen)
        trainDataset = trainDataset.shuflle(trainLen)
        return trainDataset, testDataset

    def _parseData(input, label):
        image = tf.read_file(input)
        image = decode_jpeg(image, channels=3)
        image = convert_image_dtype(image, tf.float32)
        if(input in self.maskFiles):
            imageMask = tf.read_file(self.maskDir + input)
            imageMask = decode_jpeg(imageMask, channels=3)
            imageMask = convert_image_dtype(imageMask, tf.float32)
            image = tf.math.add(image, imageMask)
            oneTensor = tf.ones_like(image)
            image = tf.math.subtract(image, oneTensor)
            image = tf.clip_by_value(image, 0.0, 1.0)
        image = resize_images(image, [self.imgSize, self.imgSize])
        return image, label

    def _preprocess(input, label):
        image = random_brightness(input, max_delta=self.maxDelta)
        image = random_saturation(
            input, lower=self.lowerSat, upper=self.upperSat)
        image = tf.clip_by_value(image, 0.0, 1.0)
