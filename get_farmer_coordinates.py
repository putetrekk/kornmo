import requests
import pandas
import urllib.parse
import csv
import keys


class Farm:
    orgnr = None
    address = None
    lat = None
    lng = None
    ele = None


def create_farm(orgnr, address):
    farm_info = Farm()
    farm_info.orgnr = orgnr
    farm_info.address = address
    farms.append(farm_info)


TERRAIN_API_URL = 'https://api.mapbox.com/v4/mapbox.mapbox-terrain-v2/tilequery/'
GEOCODING_API_URL = 'https://api.mapbox.com/geocoding/v5/mapbox.places/'


if __name__ == '__main__':
    farms = []

    df = pandas.read_csv('data/farmers_with_address.csv')

    for row in df.itertuples():
        create_farm(row[1], row[7])

    payload = {
        'access_token': keys.MAPBOX_API_KEY,
        'country': 'NO',
    }

    i = 0
    for farm in farms:
        if i % 50 == 0:
            print(i)
            print(farm.address)
        encoded_address = urllib.parse.quote(farm.address)
        r = requests.get(f"{GEOCODING_API_URL}{encoded_address}.json", params=payload)
        json = r.json()
        if r.status_code != requests.codes.ok or len(json['features']) == 0:
            continue
        farm.lng = json['features'][0]['center'][0]
        farm.lat = json['features'][0]['center'][1]
        i += 1

    with open('data/farmer_coordinates.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['orgnr', 'lng', 'lat'])
        for farm in farms:
            writer.writerow([farm.orgnr, farm.lng, farm.lat])
