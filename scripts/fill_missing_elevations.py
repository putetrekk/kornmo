import pandas
import requests

if __name__ == '__main__':
    df = pandas.read_csv('../data/farmer_elevation.csv')
    elevations_by_commune = df.groupby('commune_id').median()
    elevations_by_commune: pandas.DataFrame = elevations_by_commune['elevation']

    def get_missing_elevation(k: pandas.Series):
        if k.isnull()['elevation']:
            print(f"farmer in {k['commune']} is missing elevation!")
            return elevations_by_commune.loc[k['commune_id']]
        else:
            return k['elevation']


    df['elevation'] = df.apply(get_missing_elevation, axis=1)
    df.to_csv('data/farmer_elevation_plus.csv')
