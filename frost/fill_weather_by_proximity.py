import pandas as pd
import sys

from utils import get_weather_file_path, distance


def assign_to_farmer_and_fill_by_proximity(start_date, end_date, weather_type):
	# Weather station measurements tends to have gaps or holes in their readings,
	# this script will fill out these holes by looking at the 2nd, 3rd, 4th etc closest measurement to fill
	# these gaps, while assigning the weather to a farmer.
	processed_weather_data_path = get_weather_file_path(start_date, end_date, weather_type, 'processed')
	processed_weather_data = pd.read_csv(processed_weather_data_path)

	stations_df = pd.read_csv('data/frost_sources.csv')

	stations_with_weather_df = pd.merge(stations_df, processed_weather_data, left_on='id', right_on='station_id')

	columns_to_keep = ['id', 'lng', 'lat'] + list(filter(lambda x: x.startswith('day_'), stations_with_weather_df.columns.tolist()))
	stations_df = stations_with_weather_df[columns_to_keep]

	farmers = pd.read_csv('../data/farmer_elevation.csv').drop(columns=['Unnamed: 0']).dropna()

	farmers_with_weather = []

	number_of_farmers = farmers.shape[0]
	new_file_path = get_weather_file_path(start_date, end_date, weather_type, 'by_proximity')
	print("---- Assign weather to farmers by proximity ----")
	print(f"Number of farmers: {number_of_farmers}")
	print(f"Saving results to {new_file_path}")
	print(f"Status:")
	for index, farmer in farmers.iterrows():
		farmer_coordinates = (farmer.lat, farmer.lng)
		stations_with_distance_df = stations_df.copy()
		stations_with_distance_df['ws_distance'] = stations_with_distance_df.apply(
			lambda ws: distance(farmer_coordinates, (ws.lat, ws.lng)), axis=1)
		# Find the closest station, order by distance and ffill, keeping the top-most row.
		farmer_weather_df = stations_with_distance_df.sort_values(by=['ws_distance']).head(10).fillna(
			method='bfill').head(1)
		farmer_weather_df['orgnr'] = farmer.orgnr

		if not farmer_weather_df.dropna().empty:
			farmer_weather_df = farmer_weather_df.drop(['lng', 'lat'], axis=1)
			closest_station_id = farmer_weather_df.iloc[0].id
			missing_measurements_on_closest_station = \
			stations_df.loc[stations_df['id'] == closest_station_id].isnull().sum(axis=1).tolist()[0]
			farmer_weather_df['missing_measurements_from_closest_ws'] = missing_measurements_on_closest_station
			farmer_weather_df = farmer_weather_df.rename(columns={'id': 'ws_id', 'lng': 'ws_lng', 'lat': 'ws_lat'})
			farmers_with_weather.append(farmer_weather_df)
		else:
			print(f"Farmer location or weather missing - {farmer.orgnr}: {(farmer.lat, farmer.lng)}")

		sys.stdout.write('\r')
		sys.stdout.write(f'[{"=" * (index//200)}{" " * ((number_of_farmers - index)//200)}] {round((index / number_of_farmers + 1) * 100, 1)}%')
		sys.stdout.flush()
	print("\nDone.")
	farmers_with_weather_df = pd.concat(farmers_with_weather, ignore_index=True)
	farmers_with_weather_df.to_csv(new_file_path, index=False)
