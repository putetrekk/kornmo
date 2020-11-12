from numbers import Number
from typing import Iterable, Any
from pandas import DataFrame


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

    if lower is not None and lower == upper:
        return (df - lower) / (upper - lower)

    return (df - df.min()) / (df.max() - df.min())


def split_farmers_on_type(df: DataFrame, types: Iterable[str] = None) -> DataFrame:
    """
    Split farmers into entries for each type of crop they have grown.
    :param df: The DataFrame which contain all the farmer entries with columns for each crop type.
    :param types: The crop types we'll use to create new entries, defaults to: ['bygg', 'havre', 'hvete', 'rug_og_rughvete'].
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

        # Remove entries where nothing has been produced
        crop_df = crop_df[lambda x: (x['levert'] > 0) & (x['areal'] > 0)]

        crop_df['filter'] = crop_df['levert'] / crop_df['areal']
        crop_df = filter_extremes(crop_df, 'filter')
        crop_df.drop('filter', axis=1, inplace=True)

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
