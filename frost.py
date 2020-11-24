from typing import List
import pandas as pd
from kornmo_utils import normalize


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

    def get_as_aggregated(self, grouping_duration=7, years: List[int] = None):
        if self.weather_data is None:
            self.__load_from_files(years)

        def group_columns(columns, n):
            return [columns[x:x + n] for x in range(0, len(columns), n)]

        def aggregate_cols(dataframe, col_regex, func):
            col_subset = dataframe.filter(regex=col_regex)

            # grouping_duration of 1 day basically means we don't need to aggregate any further
            if grouping_duration == 1:
                return col_subset

            grouped_cols = group_columns(col_subset.columns, grouping_duration)
            aggregated_cols = [func(dataframe[x]) for x in grouped_cols]
            return pd.concat(aggregated_cols, axis=1)

        min_temp = aggregate_cols(self.weather_data, 'temperature_day_[0-9]+_min', lambda df: df.min(axis=1)) \
            .add_prefix("min_temp")
        max_temp = aggregate_cols(self.weather_data, 'temperature_day_[0-9]+_max', lambda df: df.max(axis=1)) \
            .add_prefix("max_temp")
        mean_temp = aggregate_cols(self.weather_data, 'temperature_day_[0-9]+_mean', lambda df: df.mean(axis=1)) \
            .add_prefix("mean_temp")
        sum_temp = aggregate_cols(self.weather_data, 'temperature_day_[0-9]+_mean', lambda df: df.sum(axis=1)) \
            .add_prefix("sum_temp")
        sum_rain = aggregate_cols(self.weather_data, 'precipitation_day_[0-9]+', lambda df: df.sum(axis=1)) \
            .add_prefix("total_rain")

        aggregated_data = pd.concat([
            self.weather_data[['year', 'orgnr']],
            normalize(self.find_growth_start()),
            normalize(min_temp, -30, 30),
            normalize(max_temp, -30, 30),
            normalize(mean_temp, -30, 30),
            # normalize(sum_temp, -30, 30),
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

        for index, row in self.weather_data.filter(regex="temperature_day_[0-9]+_mean").iterrows():
            for i in range(len(row) - 2):
                temps = row[i:i + 3]
                if is_above_threshold(temps.values):
                    growth_start_df.at[index, 'growth_start_day'] = i + 3
                    break

        return growth_start_df['growth_start_day'].astype(int)

    def __load_from_files(self, years: List[int] = None):
        if years is None:
            years = [2017, 2018, 2019]

        print(f'Loading weather data...')

        weather_data = pd.DataFrame()

        for year in years:
            precipitation = self.__load_file('precipitation', str(year))
            temperature = self.__load_file('temperature', str(year))
            weather = precipitation.merge(temperature, on=['orgnr'])
            weather['year'] = year

            weather_data = weather_data.append(weather, ignore_index=True)

        self.weather_data = weather_data

        print(f'Weather data entries loaded: {len(self.weather_data)}')

    @staticmethod
    def __load_file(file_group: str, year: str):
        data = pd.read_csv(f'data/{file_group}_{year}-04-01-{year}-10-01.csv', index_col=0).dropna()

        col_mapper = {
            element: f'{file_group}_{element}' for element in data.columns if element.startswith("day_")
        }
        data.rename(columns=col_mapper, inplace=True)

        return data.filter(items=['orgnr', *col_mapper.values()], axis=1)
