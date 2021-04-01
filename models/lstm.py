import tensorflow as tf
from tensorflow.keras import layers, models


from .single_image_cnn import timedistributed_model as CNN


def model(timesteps=30, image_shape=(100, 100, 12)):

    cnn = CNN(image_shape, output_dim=64)

    input = layers.Input(shape=(timesteps, *image_shape))
    y = layers.TimeDistributed(cnn)(input)
    y = layers.LSTM(64, return_sequences=True)(y)
    y = layers.Flatten()(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(1)(y)

    return models.Model(inputs=input, outputs=y, name="LSTM")
