import os
import pandas as pd


def store_model_history(history_obj, model_name, time):
    dir_path = f'logs/{model_name}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    hist_df = pd.DataFrame(history_obj.history)
    hist_df['name'] = model_name
    hist_df['step'] = hist_df.index
    filename = f'{dir_path}/{model_name}_epochs_{len(hist_df)}_time_{time}.csv'
    hist_df.to_csv(filename, index=False)
