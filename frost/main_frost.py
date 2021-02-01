from fill_weather_by_proximity import assign_to_farmer_and_fill_by_proximity
from preprocess_weather import preprocess_weather
from source_weather_to_file import raw_frost_readings_to_file
from utils import WEATHER_TYPES
from weather_sources_to_file import download_frost_sources

if __name__ == '__main__':

	growth_season = 2019
	start_date = f'{growth_season}-03-01'
	end_date = f'{growth_season}-10-01'

	# use WEATHER_TYPES .TEMPERATURE or .PRECIPITATION
	precipitation = WEATHER_TYPES.PRECIPITATION

	# Get all the sources from FROST
	download_frost_sources()

	# Download the weather readings for each of the sources of a given type
	raw_frost_readings_to_file(start_date, end_date, growth_season, precipitation)

	# Process the readings
	# These files can be used for interpolation / filling in the blanks by
	# looking at the 2nd closest, etc.
	preprocess_weather(start_date, end_date, precipitation)

	assign_to_farmer_and_fill_by_proximity(start_date, end_date, precipitation)
