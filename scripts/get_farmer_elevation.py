import statistics
import pandas
import requests
from tqdm import tqdm

import keys

TERRAIN_API_URL = 'https://api.mapbox.com/v4/mapbox.mapbox-terrain-v2/tilequery/'

payload = {
    'radius': '0',
    'limit': '20',
    'dedupe': 'false',
    'geometry': 'polygon',
    'access_token': keys.MAPBOX_API_KEY,
}

if __name__ == '__main__':
    farms = pandas.read_csv('../data/matrikkelen/processed/centroid_coordinates.csv', index_col='orgnr')
    farms['elevation'] = None

    p_bar = tqdm(total=len(farms), iterable=farms.iterrows())
    for index, farm in p_bar:
        try:
            r = requests.get(f"{TERRAIN_API_URL}{farm['longitude']},{farm['latitude']}.json", params=payload)
            json = r.json()
            features = json['features']
            properties = map(lambda x: x['properties'], features)
            properties = filter(lambda x: 'ele' in x, properties)
            elevations = list(map(lambda x: x['ele'], properties))
            p_bar.set_description(f"{farm.orgnr} mean elevation: {statistics.mean(elevations)}")
            farms.loc[index, 'elevation'] = statistics.mean(elevations)
        except:
            p_bar.set_description(f"{farm.orgnr} mean elevation: ERROR ENCOUNTERED, SKIPPING")

    farms[['elevation']].to_csv('../data/map/elevations.csv')
