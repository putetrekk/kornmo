import pandas as pd
from utils import normalize


class FrostDataset:
    def __init__(self):
        self.weather_data: pd.DataFrame | None = None

    def get_weather_data(self):
        if self.weather_data is not None:
            return self.weather_data.copy(deep=True)

        try:
            self.__load_from_files()
        except FileNotFoundError:
            import get_weather
            self.__load_from_files()

        if self.weather_data is not None:
            return self.weather_data.copy(deep=True)

    def get_as_aggregated(self, grouping_duration=7):
        if self.weather_data is None:
            self.__load_from_files()

        def group_columns(columns, n):
            return [columns[x:x + n] for x in range(0, len(columns), n)]

        def aggregate_cols(dataframe, col_regex, func):
            columns = dataframe.filter(regex=col_regex).columns
            grouped_cols = group_columns(columns, grouping_duration)
            aggregated_cols = [func(dataframe[x]) for x in grouped_cols]
            return pd.concat(aggregated_cols, axis=1)

        min_temp = aggregate_cols(self.weather_data, 'min_temp', lambda df: df.min(axis=1)) \
            .add_prefix("min_temp")
        max_temp = aggregate_cols(self.weather_data, 'max_temp', lambda df: df.max(axis=1)) \
            .add_prefix("max_temp")
        mean_temp = aggregate_cols(self.weather_data, 'mean_temp', lambda df: df.mean(axis=1)) \
            .add_prefix("mean_temp")
        sum_temp = aggregate_cols(self.weather_data, 'mean_temp', lambda df: df.sum(axis=1)) \
            .add_prefix("sum_temp")
        sum_rain = aggregate_cols(self.weather_data, 'sum_precipitation_mean', lambda df: df.sum(axis=1)) \
            .add_prefix("total_rain")

        aggregated_data = pd.concat([
            self.weather_data[['year', 'orgnr']],
            normalize(self.find_growth_start()),
            normalize(min_temp, -30, 30),
            normalize(max_temp, -30, 30),
            normalize(mean_temp, -30, 30),
            # normalize(monthly_sum_temp),
            normalize(sum_rain, 0, 10),
        ], axis=1)

        return aggregated_data

    def find_growth_start(self):
        def is_above_threshold(list_of_temperatures):
            for temperature_reading in list_of_temperatures:
                if temperature_reading < 5.0:
                    return False

            return True

        growth_start_df = pd.DataFrame(self.weather_data.index)

        for index, row in self.weather_data.filter(regex="mean_temp").iterrows():
            for i in range(len(row) - 2):
                temps = row[i:i + 3]
                if is_above_threshold(temps.values):
                    growth_start_df.at[index, 'growth_start_day'] = i + 3
                    break

        return growth_start_df['growth_start_day'].astype(int)

    def __load_from_files(self):
        weather_2017 = pd.read_csv('data/2017-04-01_to_2017-10-01_weather.csv').dropna()
        weather_2018 = pd.read_csv('data/2018-04-01_to_2018-10-01_weather.csv').dropna()

        weather_2017['year'] = 2017
        weather_2018['year'] = 2018

        self.weather_data = pd.concat([weather_2017, weather_2018], ignore_index=True)
