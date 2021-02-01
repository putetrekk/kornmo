import requests
import pandas as pd


def get_all_frost_sources(frost_client_id):
	url = "https://frost.met.no/sources/v0.jsonld"
	params = {
		'country': 'NO',
		'types': 'SensorSystem'
	}
	request = requests.get(url, params, auth=(frost_client_id, ''))
	req_as_json = request.json()

	data = []
	if request.status_code == 200:
		data = req_as_json['data']
		print('Sources successfully fetched frost sources')
		print(f'Sources found: {len(data)}')
	else:
		print('Error! Returned status code %s' % request.status_code)
		print('Message: %s' % req_as_json['error']['message'])
	return pd.DataFrame(data)


def download_frost_sources_to_file(target_file_path, frost_client_key):

	frost_sources_df = get_all_frost_sources(frost_client_key).dropna(subset=['geometry'])

	# Apply lng and lat to sources as easily accessible columns
	frost_sources_df['lng'] = frost_sources_df['geometry'].apply(lambda x: x['coordinates'][0])
	frost_sources_df['lat'] = frost_sources_df['geometry'].apply(lambda x: x['coordinates'][1])

	# Save as csv.
	frost_sources_df.to_csv(target_file_path, index=False)


def download_frost_sources():
	import keys
	frost_client_key = keys.frost_client_id
	target_file_path = 'data/frost_sources.csv'


	download_frost_sources_to_file(target_file_path, frost_client_key)
	return "Done"
