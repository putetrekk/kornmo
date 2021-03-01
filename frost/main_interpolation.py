import numpy as np
import pandas
from matplotlib.tri import Triangulation, LinearTriInterpolator
from numpy.ma import MaskedArray

import utils


def get_value_of_closest_sensor(lat, lng, known_values: np.array):
    closest_distance = float("inf")
    closest_index = None
    for i in range(len(known_values)):
        distance = utils.distance((lng, lat), (known_values[i, 0], known_values[i, 1]))
        if distance < closest_distance:
            closest_index = i
            closest_distance = distance
    value = known_values[closest_index, 2]
    return value


def get_interpolated_values() -> pandas.Series:
    return pandas.Series()


def generate_interpolated_csv_for_year(year_str, weather_property):

    readings = pandas.read_csv(f"processed/{weather_property}_processed_{year_str}-03-01_to_{year_str}-10-01.csv")
    sensors = pandas.read_csv(f"frost_sources.csv", index_col="id")[['lng', 'lat']]

    readings:pandas.DataFrame = readings.join(sensors, "station_id").dropna()
    readings = readings.reset_index()

    farmers = pandas.read_csv(f"../data/matrikkelen/processed/centroid_coordinates.csv")[['orgnr', 'longitude', 'latitude']]

    n_days = 0
    while True:
        if any(f'day_{n_days}' in colname for colname in readings.columns):
            n_days += 1
        else:
            break

    reps = [None]

    if weather_property == 'temperature':
        reps = ['min', 'mean', 'max']

    for i in range(n_days):
        for rep in reps:
            known_points = np.zeros([readings.shape[0], 3])
            for index, station in readings.iterrows():
                if rep is None:
                    known_points[index] = np.array([station.lng, station.lat, station[f'day_{i}']])
                else:
                    known_points[index] = np.array([station.lng, station.lat, station[f'day_{i}_{rep}']])

            # triangulation function
            tri_fn = Triangulation(known_points[:, 0], known_points[:, 1])
            # linear triangular interpolator function
            lin_tri_fn = LinearTriInterpolator(tri_fn, known_points[:, 2])

            temps = []
            for index, farmer in farmers.iterrows():
                tempytemp: MaskedArray = lin_tri_fn(farmer.longitude, farmer.latitude)
                if tempytemp.mask is np.ma.nomask:
                    temps.append(tempytemp)
                else:
                    temps.append(get_value_of_closest_sensor(farmer.latitude, farmer.longitude, known_points))

            farmers[f'day_{i}_{rep}'] = temps
            print(f'day {i} {rep} done')

    print('saving...')
    farmers.to_csv(f"interpolated/{weather_property}_interpolated_{year_str}-03-01_to_{year_str}-10-01.csv")
    print('saved')


if __name__ == '__main__':
    for year in range(2017, 2020):
        generate_interpolated_csv_for_year(str(year), 'temperature')

    for year in range(2017, 2020):
        generate_interpolated_csv_for_year(str(year), 'precipitation')


# rasterRes = 0.2
#
# xCoords = np.arange(knownPointsPerDay[:, 0].min(), knownPointsPerDay[:, 0].max() + rasterRes, rasterRes)
# yCoords = np.arange(knownPointsPerDay[:, 1].min(), knownPointsPerDay[:, 1].max() + rasterRes, rasterRes)
# zCoords = np.zeros([xCoords.shape[0], yCoords.shape[0]])
#
# for indexX, x in np.ndenumerate(xCoords):
#     for indexY, y in np.ndenumerate(yCoords):
#         tempZ = linTriFn(x,y)
#         # filtering masked values
#         if tempZ == tempZ and x != 0 and y != 0:
#             zCoords[indexX, indexY] = tempZ
#         else:
#             zCoords[indexX, indexY] = np.nan
#
# plt.imshow(zCoords, origin='lower')
# plt.show()
