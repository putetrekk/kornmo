from pandas import DataFrame

import kornmo_utils as ku

precipitation_upper = 100

temperature_lower = -30
temperature_upper = 30


def normalize_precipitation_inputs(nn_input: DataFrame) -> DataFrame:
    for i in range(3):
        nn_input[f'{i}_masl_diff'] = ku.normalize(nn_input[f'{i}_masl_diff'], -1000, 1000)
        nn_input[f'{i}_value'] = ku.normalize(nn_input[f'{i}_value'], 0, precipitation_upper)
    return nn_input


def normalize_precipitation_actual(nn_actual: DataFrame) -> DataFrame:
    nn_actual['station_x_actual'] = ku.normalize(nn_actual['station_x_actual'], 0, precipitation_upper)
    return nn_actual


def denormalize_precipitation_prediction(nn_prediction) -> DataFrame:
    return ku.denormalize(nn_prediction, 0, precipitation_upper)


def normalize_temperature_inputs(nn_input: DataFrame) -> DataFrame:
    for i in range(3):
        nn_input[f'{i}_masl_diff'] = ku.normalize(nn_input[f'{i}_masl_diff'], -1000, 1000)
        nn_input[f'{i}_min'] = ku.normalize(nn_input[f'{i}_min'], temperature_lower, temperature_upper)
        nn_input[f'{i}_mean'] = ku.normalize(nn_input[f'{i}_mean'], temperature_lower, temperature_upper)
        nn_input[f'{i}_max'] = ku.normalize(nn_input[f'{i}_max'], temperature_lower, temperature_upper)
    return nn_input


def normalize_temperature_actual(nn_actual: DataFrame) -> DataFrame:
    nn_actual['station_x_min'] = ku.normalize(nn_actual['station_x_min'], temperature_lower, temperature_upper)
    nn_actual['station_x_mean'] = ku.normalize(nn_actual['station_x_mean'], temperature_lower, temperature_upper)
    nn_actual['station_x_max'] = ku.normalize(nn_actual['station_x_max'], temperature_lower, temperature_upper)
    return nn_actual


def denormalize_temperature_prediction(nn_prediction: DataFrame) -> DataFrame:
    return ku.denormalize(nn_prediction, temperature_lower, temperature_upper)
