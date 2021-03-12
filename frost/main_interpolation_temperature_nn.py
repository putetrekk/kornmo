import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import utils
import matplotlib.pyplot as plt
import kornmo_utils as ku

normalization_lower = -30
normalization_upper = 30


def get_k_closest(sensors: pd.DataFrame, lat, lng, k: int):
    sensors['distance'] = sensors.apply(
        lambda ws: utils.distance((lat, lng), (ws.lat, ws.lng)), axis=1)
    return sensors.sort_values(by=['distance']).head(k)


def make_dataset_entry(lat, lng, masl, min, mean, max, closest: pd.DataFrame) -> pd.Series:
    day_series = pd.Series(dtype='float64')
    station_count = 0
    for station_id, station in closest.iterrows():
        day_series[f'{station_count}_lat_diff'] = station.lat - lat
        day_series[f'{station_count}_lng_diff'] = station.lng - lng
        day_series[f'{station_count}_masl_diff'] = station.masl - masl
        day_series[f'{station_count}_min'] = station.min_val
        day_series[f'{station_count}_mean'] = station.mean_val
        day_series[f'{station_count}_max'] = station.max_val
        station_count += 1
    day_series['station_x_min'] = min
    day_series['station_x_mean'] = mean
    day_series['station_x_max'] = max

    return day_series


def get_weather_dataset() -> pd.DataFrame:
    try:
        return pd.read_csv('temperatures.csv', index_col='index')
    except IOError:
        print("no existing temperatures.csv file found, will create, hold on tight!")

    frost_sources = pd.read_csv('../data/frost/frost_sources.csv', index_col=['id'])

    df: pd.DataFrame = pd.DataFrame()

    for year in range(2017, 2020):
        new_sensors: pd.DataFrame = pd.read_csv(
            f'../data/frost/processed/temperature_processed_{year}-03-01_to_{year}-10-01.csv',
            index_col='station_id').dropna()

        new_sensors = new_sensors.join(frost_sources, 'station_id')

        for day in range(214):
            print(f'day {day}, {year}')
            for station_id, station in new_sensors.iterrows():
                day_sensors = new_sensors[["lat", "lng", "masl", f"day_{day}_min", f"day_{day}_mean", f"day_{day}_max"]]
                day_sensors = day_sensors\
                    .rename(columns={f"day_{day}_min": "min_val"}) \
                    .rename(columns={f"day_{day}_mean": "mean_val"}) \
                    .rename(columns={f"day_{day}_max": "max_val"}) \
                    .drop(station_id)
                closest = get_k_closest(day_sensors, station.lat, station.lng, 3)
                series: pd.Series = make_dataset_entry(
                    station.lat,
                    station.lng,
                    station.masl,
                    station[f'day_{day}_min'],
                    station[f'day_{day}_mean'],
                    station[f'day_{day}_max'],
                    closest
                )
                df = df.append(series, ignore_index=True)

    df.to_csv('temperatures.csv', index_label='index')

    return df


def train_interpolation(train_x, train_y, val_x, val_y):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, ReLU

    input_layer = Input(shape=(len(train_x[0])), name="rudolf")
    model_x = Dense(512, activation="relu")(input_layer)
    model_x = Dense(512, activation="relu")(model_x)
    model_x = Dense(512, activation="relu")(model_x)
    model_x = Dense(128, activation="relu")(model_x)
    model_x = Dense(32, activation="relu")(model_x)

    output1 = Dense(1, name="min")(model_x)
    output2 = Dense(1, name="mean")(model_x)
    output3 = Dense(1, name="max")(model_x)

    model = Model(inputs=[input_layer], outputs=[output1, output2, output3])

    model.compile(loss=['mean_absolute_error' for _ in range(3)], optimizer=tf.keras.optimizers.Adam())
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=0.0001)
    history = model.fit(
        x=train_x,
        y=(train_y[0], train_y[1], train_y[2]),
        validation_data=(val_x, (val_y[0], val_y[1], val_y[2])),
        callbacks=[early_stopping],
        batch_size=4096,
        epochs=1000,
        verbose=2,
    )

    model.save('temperature_model.h5')

    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend()
    plt.show()

    return model


def plot(model, data_x, data_y):
    predictions = model.predict(data_x)
    df = pd.DataFrame({
        'actual_min': data_y[0].flatten(),
        'prediction_min': predictions[0].flatten(),
        'actual_mean': data_y[1].flatten(),
        'prediction_mean': predictions[1].flatten(),
        'actual_max': data_y[2].flatten(),
        'prediction_max': predictions[2].flatten()
        }
    )

    df = ku.denormalize(df, normalization_lower, normalization_upper)
    df['abs_error_min'] = abs(df['prediction_min'] - df['actual_min'])
    df['abs_error_mean'] = abs(df['prediction_mean'] - df['actual_mean'])
    df['abs_error_max'] = abs(df['prediction_max'] - df['actual_max'])
    print(f"Denormalized min MAE: {df['abs_error_min'].mean()}")
    print(f"Denormalized mean MAE: {df['abs_error_mean'].mean()}")
    print(f"Denormalized max MAE: {df['abs_error_max'].mean()}")

    plt.title('Ordered by actual temperature')

    for thing in ['mean', 'min', 'max']:
        df = df.sort_values(by=f'actual_{thing}', ignore_index=True)

        plt.plot(df[f'prediction_{thing}'], 'o', markersize=1, alpha=0.02, label=f"prediction {thing}", antialiased=True)
        plt.plot(df[f'actual_{thing}'], '--', label=f'actual temperature {thing}')

        plt.legend()

    plt.ylim(-30, 40)
    plt.show()


class NearestNeighbourModel:
    @staticmethod
    def predict(data_x):
        return data_x[:, 5:2:-1].T


if __name__ == '__main__':
    from tensorflow.keras.models import load_model

    data = get_weather_dataset().dropna()

    for i in range(3):
        data[f'{i}_masl_diff'] = ku.normalize(data[f'{i}_masl_diff'], -1000, 1000)
        data[f'{i}_min'] = ku.normalize(data[f'{i}_min'], normalization_lower, normalization_upper)
        data[f'{i}_mean'] = ku.normalize(data[f'{i}_mean'], normalization_lower, normalization_upper)
        data[f'{i}_max'] = ku.normalize(data[f'{i}_max'], normalization_lower, normalization_upper)
    data['station_x_min'] = ku.normalize(data['station_x_min'], normalization_lower, normalization_upper)
    data['station_x_mean'] = ku.normalize(data['station_x_mean'], normalization_lower, normalization_upper)
    data['station_x_max'] = ku.normalize(data['station_x_max'], normalization_lower, normalization_upper)

    y_columns = ['station_x_min', 'station_x_mean', 'station_x_max']

    train, val = train_test_split(shuffle(data), test_size=0.2)
    val, test = train_test_split(val, test_size=0.2)
    train_x = train.drop(y_columns, axis=1).to_numpy()
    train_y = train[y_columns].to_numpy().T

    val_x = val.drop(y_columns, axis=1).to_numpy()
    val_y = val[y_columns].to_numpy().T

    # model = train_interpolation(train_x, train_y, val_x, val_y)
    model = load_model('temperature_model.h5')

    plot(model, val_x, val_y)

    nearest_model = NearestNeighbourModel()

    plot(nearest_model, val_x, val_y)
