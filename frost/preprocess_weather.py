import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from utils import explode_on_column, with_dict_as_columns, with_col_as_type_datetime, append_df_to_csv, WEATHER_TYPES


def get_day_index(start_date_str, end_date_str):
	s_date = datetime.strptime(start_date_str, '%Y-%m-%d')
	e_date = datetime.strptime(end_date_str, '%Y-%m-%d')
	delta_days = (e_date - s_date).days

	return [s_date + timedelta(days=i) for i in range(delta_days)]


def get_single_observation(observations):
	# Preferably us the same offset on all, if not available just use the first.
	pt6h_obs = list(filter(lambda obs: obs['timeOffset'] == 'PT6H', observations))
	return pt6h_obs[0] if pt6h_obs else observations[0]


def with_observation_as_dict(data_df):
	data_df['observations'] = data_df['observations'].apply(get_single_observation)
	return data_df


def process_temperature_columns(station_df, station_exploded_df, date, index):
	same_date = station_exploded_df["referenceTime"].dt.date == date.date()
	measurements_of_current_date = station_exploded_df.loc[same_date]

	daily_observation_min = np.nan
	daily_observation_max = np.nan
	daily_observation_mean = np.nan

	if not measurements_of_current_date.empty:
		daily_observation_min = measurements_of_current_date.value.min()
		daily_observation_max = measurements_of_current_date.value.max()
		daily_observation_mean = round(measurements_of_current_date.value.mean(), 1)

	station_df[f'day_{index}_min'] = daily_observation_min
	station_df[f'day_{index}_max'] = daily_observation_max
	station_df[f'day_{index}_mean'] = daily_observation_mean

	return station_df


def process_precipitation_columns(station_df, station_exploded_df, date, index):
	same_date = station_exploded_df["referenceTime"].dt.date == date.date()
	measurements_of_current_date = station_exploded_df.loc[same_date]
	daily_observation_value = np.nan
	if not measurements_of_current_date.empty:
		amount_measurements = len(measurements_of_current_date)
		if amount_measurements > 1:
			print(f"More than 1 measurement on date {date}")

		# If there are several measurements of same date - Either average or take last... Take first for now..
		daily_observation_value = measurements_of_current_date['value'].iloc[0]

	station_df[f'day_{index}'] = daily_observation_value
	return station_df


def preprocess_weather(start_date, end_date, weather_type):
	new_filepath = f'data/processed/{weather_type}_processed_{start_date}_to_{end_date}.csv'
	source_filename = f'data/raw/{weather_type}_raw_{start_date}_to_{end_date}.csv'

	total_row_count = sum(1 for line in open(source_filename))
	# Process 50 rows at a time
	chunksize = 50
	chunks_processed = 0
	max_chunks = total_row_count // chunksize

	# Before starting, remove any existing data (if any)
	if os.path.exists(new_filepath):
		print(f"Removed existing file {new_filepath}")
		os.remove(new_filepath)


	print("---- Splitting weather into columns ----")
	print(f"Weather type: {weather_type}")
	print(f"Total rows to process: {total_row_count}")
	print(f"Splitting the work into chunks of {chunksize}")
	print(f"Results found in path: {new_filepath}")
	for chunk in pd.read_csv(source_filename, chunksize=chunksize):
		df = chunk.dropna().reset_index(drop=True)
		df["data"] = df["data"].apply(eval)

		stations_exploded_df = (df.pipe(explode_on_column, 'data')
		                        .pipe(with_dict_as_columns, 'data')
		                        .pipe(with_observation_as_dict)
		                        .pipe(with_dict_as_columns, 'observations')
		                        .pipe(with_col_as_type_datetime, 'referenceTime')
		                        )
		daily_index = get_day_index(start_date, end_date)
		stations_exploded_df = stations_exploded_df.groupby('station_id')

		all_station_readings = []
		for index, station_exploded_df in stations_exploded_df:
			station_df = station_exploded_df.head(1)[['station_id', 'growth_season', 'elementId', 'unit']]
			for index, date in enumerate(daily_index):
				if weather_type == 'temperature':
					station_df = process_temperature_columns(station_df, station_exploded_df, date, index)
				elif weather_type == 'precipitation':
					station_df = process_precipitation_columns(station_df, station_exploded_df, date, index)
				else:
					print(f"No processing function for {weather_type}")
			all_station_readings.append(station_df)
		seasonal_data_df = pd.concat(all_station_readings, ignore_index=True)
		append_df_to_csv(seasonal_data_df, new_filepath)

		# Status printer
		chunks_processed += 1
		rows_processed = chunks_processed * chunksize
		status_percentage = round((rows_processed / total_row_count) * 100, 1)
		sys.stdout.write('\r')
		sys.stdout.write(f'[{"=" * chunks_processed}{" " * (max_chunks - chunks_processed)}] {status_percentage}%')
		sys.stdout.flush()
	print("Done.")
