import statistics
import pandas
import requests
import keys

TERRAIN_API_URL = 'https://api.mapbox.com/v4/mapbox.mapbox-terrain-v2/tilequery/'

payload = {
    'radius': '0',
    'limit': '20',
    'dedupe': 'false',
    'geometry': 'polygon',
    'access_token': keys.MAPBOX_API_KEY,
}


def get_ele(k):
    try:
        r = requests.get(f"{TERRAIN_API_URL}{k['lng']},{k['lat']}.json", params=payload)
        json = r.json()
        features = json['features']
        properties = map(lambda x: x['properties'], features)
        properties = filter(lambda x: 'ele' in x, properties)
        elevations = list(map(lambda x: x['ele'], properties))
        print("mean elevation: " + str(statistics.mean(elevations)))
        return statistics.mean(elevations)
    except:
        return None


if __name__ == '__main__':
    farms = pandas.read_csv('data/farmers_with_address_and_coordinates.csv')
    farms['elevation'] = farms.apply(get_ele, axis=1)
    farms.to_csv('data/farmer_elevation.csv')
