import tensorflow as tf
from tensorflow.keras import layers, models

from .single_image_cnn import timedistributed_model as CNN


def model(timesteps=30, image_shape=(100, 100, 12)):

    cnn = CNN(image_shape)

    gru_input = layers.Input(shape=(timesteps, *image_shape))
    gru = layers.TimeDistributed(cnn)(gru_input)
    gru = layers.GRU(64, return_sequences=True)(gru)
    gru = layers.Flatten()(gru)
    gru = layers.Dense(64, activation=tf.nn.relu)(gru)
    gru = layers.Dense(1)(gru)

    return models.Model(inputs=gru_input, outputs=gru, name="GRU")
