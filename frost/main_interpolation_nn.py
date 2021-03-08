import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import utils
import matplotlib.pyplot as plt
import kornmo_utils as ku

normalization_upper = 100


def get_k_closest(sensors: pd.DataFrame, lat, lng, k: int):
    sensors['distance'] = sensors.apply(
        lambda ws: utils.distance((lat, lng), (ws.lat, ws.lng)), axis=1)
    return sensors.sort_values(by=['distance']).head(k)


def make_dataset_entry(lat, lng, masl, actual, closest: pd.DataFrame) -> pd.Series:
    day_series = pd.Series(dtype='float64')
    station_count = 0
    for station_id, station in closest.iterrows():
        day_series[f'{station_count}_lat_diff'] = station.lat - lat
        day_series[f'{station_count}_lng_diff'] = station.lng - lng
        day_series[f'{station_count}_masl_diff'] = station.masl - masl
        day_series[f'{station_count}_value'] = station.value
        station_count += 1
    day_series['station_x_actual'] = actual

    return day_series


def get_weather_dataset() -> pd.DataFrame:
    try:
        return pd.read_csv('numbers.csv', index_col='index')
    except IOError:
        print("no existing numbers.csv file found, will create, hold on tight!")

    frost_sources = pd.read_csv('../data/frost/frost_sources.csv', index_col=['id'])

    df: pd.DataFrame = pd.DataFrame()

    for year in range(2017, 2020):
        new_sensors: pd.DataFrame = pd.read_csv(
            f'../data/frost/processed/precipitation_processed_{year}-03-01_to_{year}-10-01.csv',
            index_col='station_id').dropna()

        new_sensors = new_sensors.join(frost_sources, 'station_id')

        for day in range(214):
            print(f'day {day}, {year}')
            for station_id, station in new_sensors.iterrows():
                day_sensors = new_sensors[["lat", "lng", "masl", f"day_{day}"]]
                day_sensors = day_sensors.rename(columns={f"day_{day}": "value"}).drop(station_id)
                closest = get_k_closest(day_sensors, station.lat, station.lng, 3)
                series: pd.Series = make_dataset_entry(
                    station.lat,
                    station.lng,
                    station.masl,
                    station[f'day_{day}'],
                    closest
                )
                df = df.append(series, ignore_index=True)

    df.to_csv('numbers.csv', index_label='index')

    return df


def train_interpolation(train_x, train_y, val_x, val_y):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Dense(units=512, activation="relu", input_dim=len(train_x[0])))
    model.add(Dropout(0.05))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.025))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=1))

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.0001))
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, min_delta=0)
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        callbacks=[early_stopping],
        batch_size=4096,
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


def plot(model, data_x, data_y):
    predictions = model.predict(data_x)
    df = pd.DataFrame({'actual': data_y.flatten(), 'prediction': predictions.flatten()})

    df = ku.denormalize(df, 0, normalization_upper)
    df['abs_error'] = abs(df['prediction'] - df['actual'])
    print(f"Denormalized MAE: {df['abs_error'].mean()}")

    plt.title('Ordered by actual precipitation')

    df = df.sort_values(by='actual', ignore_index=True)

    plt.plot(df['prediction'], 'o', markersize=1, label="prediction", alpha=0.02, antialiased=False)
    plt.plot(df['actual'], '--', label='actual precipitation')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    data = get_weather_dataset()

    for i in range(3):
        data[f'{i}_lat_diff'] = ku.normalize(data[f'{i}_lat_diff'])
        data[f'{i}_lng_diff'] = ku.normalize(data[f'{i}_lng_diff'])
        data[f'{i}_masl_diff'] = ku.normalize(data[f'{i}_masl_diff'])
        data[f'{i}_value'] = ku.normalize(data[f'{i}_value'], 0, normalization_upper)
    data['station_x_actual'] = ku.normalize(data['station_x_actual'], 0, normalization_upper)

    y_column = ['y']

    train, val = train_test_split(shuffle(data), test_size=0.2)
    val, test = train_test_split(val, test_size=0.2)
    train_x = train.drop('station_x_actual', axis=1).to_numpy()
    train_y = train['station_x_actual'].to_numpy()

    val_x = val.drop('station_x_actual', axis=1).to_numpy()
    val_y = val['station_x_actual'].to_numpy()

    model = train_interpolation(train_x, train_y, val_x, val_y)

    plot(model, val_x, val_y)
