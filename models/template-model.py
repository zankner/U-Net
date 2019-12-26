import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D
)
from tensorflow.keras import Model


class TemplateModel(Model):
    def __init__(self):
        super(model, self).__init__()
        # TODO: Initialize layers in model here

    def call(self, x):
        # TODO: Call layers on input
