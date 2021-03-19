import pandas as pd
import numpy as np
from utils import distance
from tensorflow.keras.models import load_model
import weather_interpolation_utils as wiu
from tqdm import tqdm


def get_number_of_days(readings):
    n_days = 0
    while True:
        if any(f'day_{n_days}' in col_name for col_name in readings.columns):
            n_days += 1
        else:
            return n_days


def get_k_closest(sensors: pd.DataFrame, distances: pd.DataFrame, lat, lng, masl, k: int):
    closest = distances.head(5 * k).merge(sensors, how='inner', left_on='id', right_on='station_id').head(k)

    series = pd.Series(dtype='float64')
    station_count = 0
    for station_id, station in closest.iterrows():
        series[f'{station_count}_lat_diff'] = station.lat - lat
        series[f'{station_count}_lng_diff'] = station.lng - lng
        series[f'{station_count}_masl_diff'] = station.masl - masl
        series[f'{station_count}_value'] = station.value
        station_count += 1

    return series


def generate_interpolated_precipitation_for_year(year_str):
    precipitation_model = load_model("precipitation_model.h5")
    # Tensorflow outputs some garbage on the first use, which ruins the progress bars, so let's get it over with.
    precipitation_model.predict(np.zeros(shape=(1, 12)))

    readings = pd.read_csv(f"../data/frost/processed/precipitation_processed_{year_str}-03-01_to_{year_str}-10-01.csv")
    sensors = pd.read_csv(f"../data/frost/frost_sources.csv", index_col="id")[['lng', 'lat', 'masl']]
    readings = readings.join(sensors, "station_id")
    readings = readings.reset_index()

    farmers = pd.read_csv(f"../data/matrikkelen/processed/centroid_coordinates.csv") \
        [['orgnr', 'longitude', 'latitude']]
    altitudes = pd.read_csv("../data/map/elevations.csv", index_col='orgnr')
    farmers = farmers.join(altitudes, "orgnr")

    farmer_distances = {}
    p_bar = tqdm(farmers.iterrows(), total=len(farmers))
    p_bar.set_description("Calculating distances")
    for idx, farmer in p_bar:
        farmer_sensors = sensors.copy()
        farmer_sensors['distance'] = sensors \
            .apply(lambda ws: distance((farmer.latitude, farmer.longitude), (ws.lat, ws.lng)), axis=1)
        farmer_sensors = farmer_sensors.sort_values(by=['distance'])
        farmer_distances[farmer.orgnr] = farmer_sensors[['distance']]

    n_days = get_number_of_days(readings)

    p_bar = tqdm(range(n_days))
    for day in p_bar:
        p_bar.set_description_str(f"Interpolating {year_str}, day {day} of {n_days - 1}")
        readings_for_day = readings[["station_id", "lat", "lng", "masl", f"day_{day}"]]
        readings_for_day = readings_for_day.rename(columns={f"day_{day}": "value"}).dropna()
        nn_input = pd.DataFrame()
        for index, farmer in farmers.iterrows():
            nn_input = nn_input.append(
                get_k_closest(
                    readings_for_day,
                    farmer_distances[farmer.orgnr],
                    farmer.latitude,
                    farmer.longitude,
                    farmer.elevation,
                    3),
                ignore_index=True)

        nn_input = wiu.normalize_precipitation_inputs(nn_input)
        nn_prediction = precipitation_model.predict(nn_input.to_numpy())
        nn_prediction = nn_prediction.flatten()
        farmers[f'day_{day}'] = wiu.denormalize_prediction(nn_prediction)

    print('saving...')
    farmers.to_csv(f"../data/frost/nn_interpolated/precipitation_interpolated_{year_str}-03-01_to_{year_str}-10-01.csv",
                   float_format='%.1f')
    print('saved')


if __name__ == '__main__':
    for year in range(2017, 2020):
        generate_interpolated_precipitation_for_year(str(year))
