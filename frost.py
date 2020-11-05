import pandas as pd

class Frost:
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

        return self.weather_data.copy(deep=True)

    def __load_from_files(self):
        weather_2017 = pd.read_csv('data/2017-04-01_to_2017-10-01_weather.csv').dropna()
        weather_2018 = pd.read_csv('data/2018-04-01_to_2018-10-01_weather.csv').dropna()

        weather_2017['year'] = 2017
        weather_2018['year'] = 2018

        self.weather_data = pd.concat([weather_2017, weather_2018], ignore_index=True)

    def get_as_aggregated(self, grouping_duration=7):
        if self.weather_data is None:
            self.__load_from_files()

        def group_columns(columns, n):
            return [columns[x:x + n] for x in range(0, len(columns), n)]

        def to_monthly(dataframe, col_regex, func):
            columns = dataframe.filter(regex=col_regex).columns
            grouped_cols = group_columns(columns, grouping_duration)
            aggregated_cols = [func(dataframe[x]) for x in grouped_cols]
            return pd.concat(aggregated_cols, axis=1)

        monthly_min_temp = to_monthly(self.weather_data, 'min_temp', lambda df: df.min(axis=1)) \
            .add_prefix("min_temp")
        monthly_max_temp = to_monthly(self.weather_data, 'max_temp', lambda df: df.max(axis=1)) \
            .add_prefix("max_temp")
        monthly_mean_temp = to_monthly(self.weather_data, 'mean_temp', lambda df: df.mean(axis=1)) \
            .add_prefix("mean_temp")
        monthly_sum_temp = to_monthly(self.weather_data, 'mean_temp', lambda df: df.sum(axis=1)) \
            .add_prefix("sum_temp")
        monthly_rain = to_monthly(self.weather_data, 'sum_precipitation_mean', lambda df: df.sum(axis=1)) \
            .add_prefix("total_rain")

        aggregated_data = pd.concat(
            [monthly_min_temp,
             monthly_max_temp,
             monthly_mean_temp,
             monthly_sum_temp,
             monthly_rain],
            axis=1)

        return aggregated_data