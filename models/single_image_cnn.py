import tensorflow as tf
from tensorflow.keras import layers, models


def model(input_dim=(100, 100, 12), output_dim=1):
    input_layer = layers.Input(shape=input_dim)
    y = layers.Conv2D(16, (3, 3), activation=tf.nn.relu, padding='same')(input_layer)
    y = layers.MaxPool2D((2, 2))(y)
    y = layers.Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same')(y)
    y = layers.MaxPool2D((2, 2))(y)
    y = layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same')(y)
    y = layers.MaxPool2D((2, 2))(y)
    y = layers.Flatten()(y)
    y = layers.Dense(32, activation=tf.nn.relu)(y)
    y = layers.Dense(output_dim)(y)

    return models.Model(inputs=[input_layer], outputs=[y], name="SingleImageCNN")