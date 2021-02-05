import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

def get_interpolated_values() -> pandas.Series:
    return pandas.Series()

weather = pandas.read_csv("../data/temperature_2019-04-01-2019-10-01.csv")

farmers = weather[['orgnr', 'lng_farmer', 'lat_farmer']]

weather_stations = weather.groupby(by='weather_station_id', as_index=False)
weather_stations = weather_stations.agg('median')
#weather_stations = weather_stations[['sensor_lng', 'sensor_lat', 'day_0_mean']]
#for col in weather_stations.columns:
#    print(col)

for i in range(183):
    knownPointsPerDay = np.zeros([weather.shape[0], 3])
    for index, station in weather_stations.iterrows():
        knownPointsPerDay[index] = np.array([station.sensor_lng, station.sensor_lat, station[f'day_{i}_mean']])

    # triangulation function
    triFn = Triangulation(knownPointsPerDay[:, 0], knownPointsPerDay[:, 1])
    # linear triangule interpolator funtion
    linTriFn = LinearTriInterpolator(triFn, knownPointsPerDay[:, 2])

    temps = []
    for index, farmer in farmers.iterrows():
        tempytemp = linTriFn(farmer.lng_farmer, farmer.lat_farmer)
        temps.append(tempytemp)

    farmers[f'day_{i}_mean'] = temps
    print(f'day {i} done')

print('saving...')
farmers.to_csv('INTERPOLATION_TEST.csv')
print('saved')

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
