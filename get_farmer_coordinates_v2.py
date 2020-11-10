import ast
import keys
import requests
import pandas
import urllib.parse

GEOCODING_API_URL = 'https://api.mapbox.com/geocoding/v5/mapbox.places/'

if __name__ == '__main__':
    df = pandas.read_csv('data/farmers_with_address.csv', index_col='orgnr')
    df['lat'] = ''
    df['lng'] = ''

    processed = 0
    for orgnr, row in df.iterrows():
        farm = df.loc[orgnr].copy()

        # SELECT LAST ADRESS, IT SEEMS TO BE THE MOST SPECIFIC
        if '[]' in farm['address']:
            farm['address'] = ''
        elif '[' in farm['address']:
            addresslist = ast.literal_eval(farm['address'])
            farm['address'] = addresslist[-1]

        # ENCODE SEARCH QUERY
        sok = ' '.join([
            farm['address'],
            str(farm['postal_code']),
            farm['postal_place'],
            farm['commune'],
        ])
        sok = urllib.parse.quote(sok)

        payload = {
            'access_token': keys.MAPBOX_API_KEY,
            'country': 'NO',
        }

        # REQUEST GEOCODING
        response = requests.get(f"{GEOCODING_API_URL}{sok}.json", params=payload)
        json = response.json()

        try:
            results = json['features']

            # SELECT BEST RESULT
            best_index = 0
            result = results[best_index]

            df.loc[orgnr, 'lng'] = result['center'][0]
            df.loc[orgnr, 'lat'] = result['center'][1]
        except:
            pass

        if processed % 50 == 0:
            print(processed)
        processed += 1

    df.to_csv('data/farmers_coordinates.csv')
