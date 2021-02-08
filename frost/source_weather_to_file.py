import requests
import pandas as pd
import numpy as np
import sys
from utils import append_df_to_csv, WEATHER_TYPES


def get_frost_data(frost_client_id, source_id, ref_time, element, time_resolution=''):
	# Check Frost if the source has data on given season and element
	params = {
		'sources': [source_id],
		'elements': element,
		'referencetime': ref_time
	}

	if time_resolution:
		params['timeresolutions'] = time_resolution

	observations_url = 'https://frost.met.no/observations/v0.jsonld'
	request = requests.get(observations_url, params, auth=(frost_client_id, ''))
	json = request.json()

	if request.status_code == 200:
		data = json['data']
		return data
	else:
		return np.nan


def download_raw_weather_data(weather_type, from_date, to_date, growth_season, frost_sources_df, frost_elements, time_resolution):
	import keys
	frost_client_key = keys.frost_client_id

	data_type = 'raw'
	file_path = f'data/raw/{weather_type}_{data_type}_{from_date}_to_{to_date}.csv'

	# If the
	import os
	if os.path.exists(file_path):
		print("Deleted existing file before downloading")
		os.remove(file_path)
	else:
		print("The file does not exist")

	count_sources = frost_sources_df.shape[0]
	for index, frost_source in frost_sources_df.iterrows():
		# Get weather for each station
		# Append it to a csv
		ref_time = f'{from_date}/{to_date}'
		station_id = frost_source['id']
		frost_data = get_frost_data(frost_client_key, station_id, ref_time, frost_elements, time_resolution)
		data = {'station_id': station_id, 'data': frost_data, 'growth_season': growth_season}

		df = pd.DataFrame([data])
		append_df_to_csv(df, file_path)

		sys.stdout.write('\r')
		sys.stdout.write(f'[{"=" * (index//20)}{" " * ((count_sources - index)//20)}] {round((index / count_sources) * 100, 1)}%')
		sys.stdout.flush()


def raw_frost_readings_to_file(from_date, to_date, growth_season, weather_type):
	frost_sources = pd.read_csv('data/frost_sources.csv')

	print("---- Downloading raw readings ----")
	print(f"Weather type: {weather_type}")
	if weather_type == WEATHER_TYPES.PRECIPITATION:
		precipitation_frost_elements = 'sum(precipitation_amount P1D)'
		precipitation_time_resolution = 'P1D'
		download_raw_weather_data('precipitation', from_date, to_date, growth_season, frost_sources, precipitation_frost_elements, precipitation_time_resolution)
	elif weather_type == WEATHER_TYPES.TEMPERATURE:
		temperature_frost_elements = 'air_temperature'
		temperature_time_resolutions = ''
		download_raw_weather_data('temperature', from_date, to_date, growth_season, frost_sources, temperature_frost_elements, temperature_time_resolutions)
	print("... Done.")
