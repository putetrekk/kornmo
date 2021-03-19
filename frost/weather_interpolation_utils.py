from pandas import DataFrame

import kornmo_utils as ku

precipitation_upper = 100


def normalize_precipitation_inputs(nn_input: DataFrame) -> DataFrame:
    for i in range(3):
        nn_input[f'{i}_masl_diff'] = ku.normalize(nn_input[f'{i}_masl_diff'], -1000, 1000)
        nn_input[f'{i}_value'] = ku.normalize(nn_input[f'{i}_value'], 0, precipitation_upper)
    return nn_input


def normalize_precipitation_actual(nn_actual: DataFrame) -> DataFrame:
    nn_actual['station_x_actual'] = ku.normalize(nn_actual['station_x_actual'], 0, precipitation_upper)
    return nn_actual


def denormalize_prediction(nn_prediction) -> DataFrame:
    return ku.denormalize(nn_prediction, 0, precipitation_upper)