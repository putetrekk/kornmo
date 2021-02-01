import pandas as pd
import math


def explode_on_column(data_df: pd.DataFrame, col_key: str):
    """
    Explodes on column by given key, and returns the dataframe.
    """
    return data_df.explode(col_key)


def with_dict_as_columns(data_df: pd.DataFrame, col_key: str):
    """
    Makes a column containing a dict into several columns from the dict, and returns the dataframe
    """
    return pd.concat([data_df.drop([col_key], axis=1), data_df[col_key].apply(pd.Series)], axis=1)


def with_col_as_type_datetime(data_df: pd.DataFrame, col_key: str):
    """
    Convert a given column into type datetime, and returns the dataframe
    """
    data_df[col_key] = pd.to_datetime(data_df[col_key])
    return data_df


def append_df_to_csv(df, csv_file_path, sep=","):
    """
    Safely append to a csv file. For things such as chunking with pandas etc.
    Creates a file if it does not exist. Remember to cleanup / remove files inbetween runs.

    from stackoverflow
    """
    import os
    if not os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csv_file_path, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csv_file_path, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csv_file_path, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csv_file_path, mode='a', index=False, sep=sep, header=False)


def get_weather_file_path(start_date, end_date, weather_type, file_type):
    return f'data/{file_type}/{weather_type}_{file_type}_{start_date}_to_{end_date}.csv'


def distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return round(d, 2)


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DotDict ' + dict.__repr__(self) + '>'


WEATHER_TYPES = DotDict({
    'TEMPERATURE': 'temperature',
    'PRECIPITATION': 'precipitation'
})
