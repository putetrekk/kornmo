import pandas as pd


def get_farmer_elevation():
    csv = pd.read_csv('data/farmer_elevation.csv')

    return csv[['orgnr', 'lat', 'elevation']].copy(deep=True)