import tensorflow as tf
from tensorflow.keras import layers, models


from .single_image_cnn import model as CNN


def model(timesteps=30, image_shape=(100, 100, 12)):

    cnn = CNN(image_shape)

    lstm_input = layers.Input(shape=(timesteps, 90, 90, 13))
    lstm = layers.TimeDistributed(cnn)(lstm_input)
    lstm = layers.LSTM(256, return_sequences=True)(lstm)
    lstm = layers.LSTM(128, return_sequences=True)(lstm)
    lstm = layers.Flatten()(lstm)
    lstm = layers.Dense(256, activation=tf.nn.relu)(lstm)
    lstm = layers.Dense(1)(lstm)

    return models.Model(inputs=lstm_input, outputs=lstm, name="LSTM")
