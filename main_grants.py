from kornmo import KornmoDataset
from frost import FrostDataset
import kornmo_utils as ku
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from dense_model import train_simple_dense
from visualize import plot
import numpy as np
import pandas as pd
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_levert_per_tilskudd(data_df: pd.DataFrame):
    df = data_df.copy()
    df['levert_per_tilskudd'] = df['levert'] / df['areal_tilskudd']
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


if __name__ == '__main__':
	kornmo = KornmoDataset()
	frost = FrostDataset()

	data = kornmo.get_legacy_data().pipe(ku.sum_and_one_hot_grain)

	weather_data = frost.get_as_aggregated(7, [2013, 2014, 2015, 2016, 2017, 2018, 2019])
	data = data.merge(weather_data, on=['year', 'orgnr'])
	y_column_key = 'levert_per_tilskudd'

	normalize_cols = ['areal_tilskudd', 'levert_per_tilskudd', 'levert', 'lat', 'elevation', 'growth_start_day']
	data = data\
		.pipe(ku.merge_with_elevation_data)\
		.pipe(get_levert_per_tilskudd)\
		.pipe(ku.filter_extremes, 'levert')\
		.pipe(ku.filter_extremes, 'areal_tilskudd')\
		.pipe(ku.filter_extremes, 'levert_per_tilskudd')\
		.pipe(ku.normalize_by_keys, normalize_cols)\
		.pipe(ku.one_hot_column, 'komnr')\
		.pipe(ku.one_hot_column, 'year')\
		.pipe(lambda df_: df_.drop(['year', 'komnr'], axis=1))

	# Split into training and validation data
	y_column = ['levert_per_tilskudd']
	remove_from_training = ['orgnr', 'levert'] + y_column

	train, val = train_test_split(shuffle(data), test_size=0.2)
	val, test = train_test_split(val, test_size=0.2)
	train_x = train.drop(remove_from_training, axis=1).to_numpy()
	train_y = train[y_column].to_numpy()

	val_x = val.drop(remove_from_training, axis=1).to_numpy()
	val_y = val[y_column].to_numpy()

	model = train_simple_dense(train_x, train_y, val_x, val_y)
	plot(model, val_x, val_y)
