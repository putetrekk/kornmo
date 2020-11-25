import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot(model, data_x, data_y):
    predictions = model.predict(data_x)
    df = pd.DataFrame({'relative production': data_y.flatten(), 'prediction': predictions.flatten()})

    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 12), squeeze=True)
    __plot_prediction_vs_y_1(df, ax1)
    __plot_prediction_vs_y_2(df, ax2)

    plt.show()


def __plot_prediction_vs_y_1(df, ax):
    ax[0].title.set_text('Farmers ordered by their relative production')
    ax[1].title.set_text('Farmers ordered by their relative production')

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
