import tensorflow as tf
from tensorflow.keras import layers, models

from .single_image_cnn import model as CNN


def model(timesteps=30, image_shape=(100, 100, 12)):
    timesteps = 30

    cnn = CNN(image_shape)

    gru_input = layers.Input(shape=(timesteps, 90, 90, 13))
    gru = layers.TimeDistributed(cnn)(gru_input)
    gru = layers.GRU(256, return_sequences=True)(gru)
    gru = layers.GRU(128, return_sequences=True)(gru)
    gru = layers.Flatten()(gru)
    gru = layers.Dense(256, activation=tf.nn.relu)(gru)
    gru = layers.Dense(1)(gru)

    return models.Model(inputs=gru_input, outputs=gru, name="GRU")
