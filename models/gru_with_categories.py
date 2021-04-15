import tensorflow as tf
from tensorflow.keras import layers, models

from .single_image_cnn import timedistributed_model as CNN


def model(timesteps=30, image_shape=(90, 90, 13)):

    timeseries_input = layers.Input(shape=(timesteps, *image_shape), name="cnn_input")
    feature_input = layers.Input(shape=(5,), name="feature_input")
    
    cnn = CNN(image_shape, 64)
    model = layers.TimeDistributed(cnn)(timeseries_input)
    repeat = layers.RepeatVector(30)(feature_input)
    model = layers.Concatenate(axis=2)([model, repeat])

    model = layers.GRU(128, return_sequences=True)(model)
    model = layers.Flatten()(model)
    model = layers.Dense(128, activation=tf.nn.relu)(model)
    model = layers.Dense(1)(model)

    return models.Model(inputs=[timeseries_input, feature_input], outputs=model, name="GRU")