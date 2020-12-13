

def train_simple_dense(train_x, train_y, val_x, val_y):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential


    model = Sequential()
    model.add(Dense(units=256, activation="tanh", input_dim=len(train_x[0])))
    model.add(Dropout(0.1))
    model.add(Dense(units=64, activation="tanh"))
    model.add(Dropout(0.25))
    model.add(Dense(units=64, activation="tanh"))
    model.add(Dense(units=1))

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.0001))
    model.summary()

    #model.load_weights("testweights")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=0.0001)
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        callbacks=[early_stopping],
        batch_size=512,
        epochs=1000,
        verbose=2,
    )

    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend()
    plt.show()

    return model
