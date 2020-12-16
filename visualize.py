import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from visualize_utils import make_image_path


def plot(model, data_x, data_y):
    predictions = model.predict(data_x)
    df = pd.DataFrame({'relative production': data_y.flatten(), 'prediction': predictions.flatten()})

    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(8, 8), squeeze=True)
    __plot_prediction_vs_y_1(df, ax1)
    __plot_prediction_vs_y_2(df, ax2)

    plt.show()


def __plot_prediction_vs_y_1(df, ax):
    ax[0].title.set_text('Ordered by relative production (kg per grant)')
    ax[1].title.set_text('Ordered by relative production (kg per grant)')

    df = df.sort_values(by='relative production', ignore_index=True)

    r_window = 50
    rolling_df = df[['prediction']].rolling(r_window, min_periods=1)

    med_a = rolling_df.median()
    upper_a = rolling_df.max()
    lower_a = rolling_df.min()

    pred = pd.concat([lower_a, med_a, upper_a], axis=1)[::r_window].join(df['relative production'])
    sns.lineplot(data=pred[['prediction', 'relative production']], ax=ax[1])

    ax[0].plot(df['prediction'], 'o', markersize=1, label="prediction")
    ax[0].plot(df['relative production'], '--', label='relative production')
    ax[0].legend()


def __plot_prediction_vs_y_2(df, ax):
    ax[0].title.set_text('Ordered by prediction')
    ax[1].title.set_text('Ordered by prediction')

    df = df.sort_values(by=['prediction'], ignore_index=True)

    r_window = 50
    rolling_df = df[['relative production']].rolling(r_window, min_periods=1)

    med_a = rolling_df.median()
    upper_a = rolling_df.max()
    lower_a = rolling_df.min()

    pred = df[['prediction']].join(pd.concat([lower_a, med_a, upper_a], axis=1))[::r_window]
    sns.lineplot(data=pred[['relative production', 'prediction']], ax=ax[1])

    ax[0].plot(df['relative production'], 'o', markersize=1, label='relative production')
    ax[0].plot(df['prediction'], '--', label="prediction")
    ax[0].legend()


def change_grown_type(farmer_dataframe, all_combos):
    for index, row in farmer_dataframe.iterrows():
        combo_to_use = all_combos[index]
        for key in combo_to_use:
            farmer_dataframe.at[index, key] = combo_to_use[key]
    return farmer_dataframe


translate_types = {
    'bygg': 'barley',
    'havre': 'oats',
    'hvete': 'wheat',
    'rug_og_rughvete': 'rye and ryewheat'
}


def translate_grain_types(grain_types):
    translated_grain_types = []

    for grain_type in grain_types:
        try:
            translated_grain_types.append(translate_types[grain_type])
        except KeyError:
            translated_grain_types.append(grain_type)
    return translated_grain_types


def generate_alternative_outcomes(data_df, model, y_column, remove_from_training, area_type, generate_limit=5):
    farmers = data_df.reset_index(drop=True).head(generate_limit)
    test_cases = []

    grain_types = ['bygg', 'havre', 'hvete', 'rug_og_rughvete']

    all_combos = []
    my_dict = {}
    for gt in grain_types:
        my_dict[gt] = 0

    for i in range(len(grain_types)):
        current_dict = my_dict.copy()
        current_dict[grain_types[i]] = 1
        all_combos.append(current_dict)

    for index, row in farmers.iterrows():
        planted_grains = []
        for grain_type in grain_types:
            if (row[grain_type]):
                planted_grains.append(grain_type)

        row_as_df = pd.DataFrame.from_dict([row.to_dict()])
        alternatives_to_farmers_df = pd.concat([row_as_df] * len(grain_types), ignore_index=True)

        alternatives_to_farmers_df = change_grown_type(alternatives_to_farmers_df, all_combos)
        farmer_data = {
            'orgnr': row["orgnr"],
            'planted': ", ".join([str(elem) for elem in planted_grains]),
            'y': row[y_column[0]],
            'alternatives': alternatives_to_farmers_df
        }

        test_cases.append(farmer_data)

    for farmer in test_cases:
        alternatives = farmer['alternatives']
        planted = farmer['planted']
        pred_this = alternatives.drop(remove_from_training, axis=1)
        pred = model.predict(pred_this)


        labels = grain_types + [planted]
        labels = translate_grain_types(labels)
        x_pos = [i for i, _ in enumerate(labels)]
        plot_data = pred.flatten().tolist()
        # Add the Actual performance of the farmer
        all_data = plot_data + [farmer['y']]

        plt.bar(x_pos, all_data,
                color=[(0.2, 0.4, 0.6, 0.6), (0.2, 0.4, 0.6, 0.6), (0.2, 0.4, 0.6, 0.6), (0.2, 0.4, 0.6, 0.6), 'green'])

        plt.xlabel("Type")
        plt.ylabel(f"Normalized kg delivered/ {area_type}")
        plt.title(f"Prediction vs Actual")
        plt.xticks(x_pos, labels)
        plt.savefig(make_image_path(f'areal_{farmer["orgnr"]}'), format='png')
        plt.show()