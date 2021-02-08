from numbers import Number
from typing import Iterable, Any, List
from pandas import DataFrame
import pandas as pd
from sklearn import preprocessing

import pandas as pd

def flatmap(f, xs):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys


def filter_extremes(df: DataFrame, column: Any, lower: float = None, upper: float = None) -> DataFrame:
    """
    Removes values which are more than 2 standard deviations away from the mean.
    :param df: The DataFrame to filter .
    :param column: The column in the dataframe which is filtered on.
    :param lower: All rows with a value below this is removed
    :param upper: All rows with a value below this is removed
    :return: A new DataFrame with extreme values of the given column filtered out.
    """

    mean = df[column].mean()
    std = df[column].std()
    if lower is None:
        lower = mean - 2 * std
    if upper is None:
        upper = mean + 2 * std

    filtered_data = df[lambda x: (x[column] > lower) & (x[column] < upper)]
    filtered_data.reset_index(drop=True, inplace=True)
    return filtered_data


def normalize(df, lower: float = None, upper: float = None) -> DataFrame:
    """
    :param df: The DataFrame where all columns will be normalized.
    :param lower: if present, together with upper, this value will correspond to the normalized value of 0.
    :param upper: if present, together with lower, this value will correspond to the normalized value of 1.
    :return: The new normalized DataFrame
    """

    if lower is None:
        lower = df.min()
    if upper is None:
        upper = df.max()

    return (df - lower) / (upper - lower)


def normalize_by_key(df, col_key):
    if col_key in df:
        df[col_key] = preprocessing.MinMaxScaler().fit_transform(df[[col_key]])
    else:
        print(f'Could not find {col_key} in DataFrame')
    return df


def normalize_by_key_and_values(df, col_key, min_val, max_val):
    if col_key in df:
        df[col_key] = (df[col_key] - min_val) / (max_val - min_val)
    else:
        print(f'Could not find {col_key} in DataFrame')
    return df


def standardize_column_by_key(df, col_key):
    if col_key in df:
        df[col_key] = preprocessing.StandardScaler().fit_transform(df[[col_key]])
    else:
        print(f'Could not find {col_key} in DataFrame')
    return df


def normalize_by_keys(df, col_keys: List[str]):
    for key in col_keys:
        df = normalize_by_key(df, key)
    return df


def get_historical_production(kornmo, years: List[int] = None, look_back_years: int = 4) -> DataFrame:
    """
    Creates a DataFrame with all farmers for each year in 'years' (default: 2017-2019),
    with each farmers production numbers for previous years, looking back the number of years specified.
    :param kornmo: An instance of a KornmoDataset
    :param years: The dataframe will have one row for each of these years per farmer
    :param look_back_years: Each row will contain this many years of production numbers
    """

    import pandas as pd
    from functools import reduce

    if years is None:
        years = [2017, 2018, 2019]

    deliveries_by_year = kornmo.get_historical_deliveries_by_year()

    data = DataFrame()

    for year in years:
        dataframes = [df.copy() for y, df in deliveries_by_year.items() if year - look_back_years <= y < year]

        for index, x in enumerate(dataframes):
            x.drop('year', axis=1, inplace=True)
            x.columns = ['orgnr', f'bygg_sum_{index}', f'hvete_sum_{index}',
                         f'havre_sum_{index}', f'rug_og_rughvete_sum_{index}']

        history_data = reduce(lambda left, right: pd.merge(left, right, on=['orgnr'], how='outer'), dataframes)

        history_data.insert(0, 'year', year)

        data = data.append(history_data, ignore_index=True)

    data_index_cols = data.filter(items=['orgnr', 'year'], axis=1)
    data_cols = data.filter(items=[col for col in data.columns if col not in ['orgnr', 'year']])

    return data_index_cols.merge(normalize(data_cols.fillna(0), 0, 10000), left_index=True, right_index=True)


def split_farmers_on_type(df: DataFrame, types: Iterable[str] = None, remove_outliers=True) -> DataFrame:
    """
    Split farmers into entries for each type of crop they have grown.
    :param df: The DataFrame which contain all the farmer entries with columns for each crop type.
    :param types: The crop types we'll use to create new entries, defaults to: ['bygg', 'havre', 'hvete', 'rug_og_rughvete'].
    :param remove_outliers: Whether to remove outliers in the dataset for each crop type
    :return: A new DataFrame where every entry is split up into entries for each crop type.
    """

    from pandas import concat
    from numpy import array

    if types is None:
        types = ['bygg', 'havre', 'hvete', 'rug_og_rughvete']

    crop_columns = list(map(lambda x: (f'{x}_sum', f'{x}_areal'), types))

    data = DataFrame()

    for index, crop_type in enumerate(types):
        crop_sum, crop_area = crop_columns[index]

        crop_df = df.filter(items=['year', 'orgnr', crop_sum, crop_area])
        crop_df.rename(columns={crop_sum: 'levert', crop_area: 'areal'}, inplace=True)

        # Set type
        crop_df[crop_type] = 1

        if remove_outliers:
            # Remove entries where nothing has been produced
            crop_df = crop_df[lambda x: (x['levert'] > 0) & (x['areal'] > 0)]

            crop_df['filter'] = crop_df['levert'] / crop_df['areal']
            crop_df = filter_extremes(crop_df, 'filter')
            crop_df.drop('filter', axis=1, inplace=True)

            crop_df.reset_index(drop=True, inplace=True)
            data = concat([data, crop_df], axis=0, ignore_index=True)

        else:
            crop_df.reset_index(drop=True, inplace=True)
            data = concat([data, crop_df], axis=0, ignore_index=True)

    data.fillna(0, inplace=True)

    df_clean = df.drop(array(crop_columns).flatten(), axis=1)
    return df_clean.merge(data, on=['year', 'orgnr'])


def split_train_validation(df: DataFrame, val_split: Number = 0.2):
    """
    Split the data into training and validation
    """

    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle

    train, val = train_test_split(shuffle(df), test_size=val_split)

    return train, val


def prepare_train_validation(df: DataFrame, y_col: Any, val_split: Number = 0.2):
    """
    Split the data into training and validation and prepare the data for the neural network
    """

    train, val = split_train_validation(df, val_split=val_split)

    train_x = train.drop(y_col, axis=1).to_numpy()
    train_y = train[y_col].to_numpy()

    val_x = val.drop(y_col, axis=1).to_numpy()
    val_y = val[y_col].to_numpy()

    return train_x, train_y, val_x, val_y


def add_noise(df: DataFrame, a=-1, b=1, method='add') -> DataFrame:
    """
    Add noise to all rows in a DataFrame
    :param df: The DataFrame we want to add noise to
    :param a: lower limit of random number that we'll use
    :param b: upper limit of random number that we'll use
    :param method: 'add' or 'mul'. The method used to add noise: add or multiply every value with a random value
    :return: A new DataFrame with random numbers between 'a' and 'b' added to every row
    """
    import numpy as np

    rand_vector = (b - a) * np.random.random_sample((len(df), len(df.columns))) + a
    print(rand_vector)

    if method == 'add':
        return df.add(rand_vector, axis=0)
    if method == 'mul':
        return df.mul(rand_vector, axis=0)

    raise AssertionError("Method must be either 'add' or 'mul'")


def sum_and_one_hot_grain(df: DataFrame) -> DataFrame:
    """
    Sums all gratins into column 'levert'
    Since we do not know how much area is used by each grain type, one hot them.
    """
    only_grain_farmers = df.copy()
    only_grain_farmers = only_grain_farmers.drop(columns=['erter_sum', 'oljefro_sum'])
    only_grain_farmers['levert'] = only_grain_farmers.filter(regex="_sum").sum(axis=1)

    grain_colums = only_grain_farmers.filter(regex="_sum").keys()

    for grain_column in grain_colums:
        only_grain_farmers[grain_column] = only_grain_farmers[grain_column].apply(lambda x: 1 if x > 0 else 0)

    rename_cols = {}
    for key in grain_colums:
        rename_cols[key] = key.split('_sum')[0]

    only_grain_farmers = only_grain_farmers.rename(columns=rename_cols)
    return only_grain_farmers


def one_hot_column(data_df: DataFrame, col_key: str):
    df = data_df.copy()
    df[col_key] = pd.Categorical(df[col_key])
    dfDummies = pd.get_dummies(df[col_key], prefix=col_key)
    df = pd.concat([df, dfDummies], axis=1)

    return df


def merge_with_elevation_data(df_to_merge):
    df = df_to_merge.copy()
    elevation_data = pd.read_csv('data/farmer_elevation.csv').dropna()
    elevation_data = elevation_data[['orgnr', 'lat', 'elevation']]
    return df.merge(elevation_data, on=['orgnr'])


def convert_to_new_gaardsnr(df: DataFrame, komnr_col, gaardsnr_col, bruksnr_col, festenr_col):
    mapper = pd.read_csv("data/geonorge/nye_gaards_og_bruksnummer.csv")

    old_cols = ['old_kommunenr','old_gaardsnummer','old_bruksnummer','old_festenummer']
    new_cols = ['new_kommunenr','new_gaardsnummer','new_bruksnummer','new_festenummer']

    original_cols = [komnr_col, gaardsnr_col, bruksnr_col, festenr_col]

    return mapper.merge(
        df,
        left_on=old_cols,
        right_on=original_cols)\
      .drop(columns=old_cols + original_cols)\
      .rename(columns={ new: original_cols[i] for i, new in enumerate(new_cols) })