import requests
import pandas as pd
import keys


def get_all_frost_sources():
    url = "https://frost.met.no/sources/v0.jsonld"
    request = requests.get(url, auth=(keys.FROST_CLIENT_ID, ''))
    req_as_json = request.json()
    data = []
    if request.status_code == 200:
        data = req_as_json['data']
        print('Sources successfully fetched frost sources')
        print(f'Sources found: {len(data)}')
    else:
        print('Error! Returned status code %s' % request.status_code)
        print('Message: %s' % req_as_json['error']['message'])
    return pd.DataFrame(data)


def clean_frost_sources(frost_sources_dataframe):
    frost_sources_dataframe['municipalityId'] = frost_sources_dataframe['municipalityId'].fillna(value=0)
    frost_sources_dataframe['municipalityId'] = frost_sources_dataframe['municipalityId'].astype(int)

    return frost_sources_dataframe


commune_without_sensors = []


def assign_farmers_to_weather_sensor(farmers_df, sourcesDf):
    success_station = 0
    fail_station = 0
    farmers = []

    for index, row in farmers_df.iterrows():
        orgnr = row['orgnr']
        name = row['name']
        commune = row['commune']
        commune_id = row['brreg_commune_id']

        found_source = sourcesDf.loc[sourcesDf['municipalityId'] == commune_id]

        if not found_source.empty:
            success_station += 1

            frost_sensor_id = found_source['id'].values

            # Make it frost API friendly by using format: 'id1, id2, id3' string
            frost_sensor_id = ', '.join(frost_sensor_id)

            farmers_df.at[index, 'frost_sensor_ids'] = frost_sensor_id
        else:
            commune_without_sensors.append(commune)
            fail_station += 1

    print(f'Number of farmers in commune with weather sensors: {success_station}. Without sensors: {fail_station}.')
    return farmers_df.dropna(subset=['frost_sensor_ids'])


# Sample Get Observations
# Sources must be in format : 'sourceid1, sourceid2, sourceid3' etc (string)
# Elements must be in format: 'max(air_temperature P1D), min(air_temperature P1D)'
# Referencetime must be in format: '2018-04-01/2018-04-05'
def get_observations(params):
    url = 'https://frost.met.no/observations/v0.jsonld'
    request = requests.get(url, params, auth=(keys.FROST_CLIENT_ID, ''))
    json = request.json()

    # Check if the request worked, print out any errors
    if request.status_code == 200:
        data = json['data']
        return data
    else:
        return []


def apply_temperature(farmers_df, ref_time):
    print("Applying temperatures....")
    grouped_df = [x for _, x in farmers_df.groupby('commune')]

    elements = 'air_temperature'

    params = {
        'elements': elements,
        'referencetime': ref_time,
    }

    error_amount = 0
    error_communes = []
    for group_index in range(len(grouped_df)):

        if group_index % 10 == 0:
            print(f'{group_index} of {len(grouped_df)} done.')
        sensor_ids = grouped_df[group_index]['frost_sensor_ids'].iloc[0]
        commune = grouped_df[group_index]['commune'].iloc[0]
        params['sources'] = sensor_ids

        readings = get_observations(params)

        for index, res in enumerate(readings):
            # Might need to handle cases with no, or several observations.
            readings[index].update(res['observations'][0])

        readings_as_df = pd.DataFrame(readings)
        try:
            readings_as_df['referenceTime'] = pd.to_datetime(readings_as_df['referenceTime'])

            list_of_days = [x for _, x in readings_as_df.groupby(pd.Grouper(key="referenceTime", freq="D"))]

            for i in range(len(list_of_days)):
                grouped_df[group_index][f'day{i}_mean_temp'] = list_of_days[i]['value'].mean()
                grouped_df[group_index][f'day{i}_max_temp'] = list_of_days[i]['value'].max()
                grouped_df[group_index][f'day{i}_min_temp'] = list_of_days[i]['value'].min()
        except:
            error_amount += 1
            error_communes.append(commune)

    print("Done applying temperatures")
    print(f'errors found: {error_amount}')
    print(f'communes with errors: {error_communes}')
    return pd.concat(grouped_df)


def apply_precipitation(farmers_df, ref_time):
    print("----------------- Applying precipitation ---------------------")
    grouped_df = [x for _, x in farmers_df.groupby('commune')]

    elements = 'sum(precipitation_amount P1D)'

    params = {
        'elements': elements,
        'referencetime': ref_time,
        'timeresolutions': 'P1D'
    }

    error_amount = 0
    error_communes = []
    for group_index in range(len(grouped_df)):
        if group_index % 10 == 0:
            print(f'{group_index} of {len(grouped_df)} done.')
        sensor_ids = grouped_df[group_index]['frost_sensor_ids'].iloc[0]
        commune = grouped_df[group_index]['commune'].iloc[0]

        params['sources'] = sensor_ids
        readings = get_observations(params)

        for index, res in enumerate(readings):
            # Maybe need to handle cases with *several* observations (per sensor).
            readings[index].update(res['observations'][0])

        readings_as_df = pd.DataFrame(readings)
        try:
            readings_as_df['referenceTime'] = pd.to_datetime(readings_as_df['referenceTime'])
            list_of_days = [x for _, x in readings_as_df.groupby(pd.Grouper(key="referenceTime", freq="D"))]
            for i in range(len(list_of_days)):
                grouped_df[group_index][f'day{i}_sum_precipitation_mean'] = list_of_days[i]['value'].mean()
        except:
            error_amount += 1
            error_communes.append(commune)

    print("Done applying precipitation")
    print(f'errors found: {error_amount}')
    print(f'communes with errors: {error_communes}')
    return pd.concat(grouped_df)


def save_to_file(df, start_date, end_date, path):
    print("Saving dataframe....")
    file_path = f"{path}/{start_date}_to_{end_date}_weather.csv"
    df.to_csv(file_path)
    print("Done.")
    print(f'File path = {file_path}')
    return df


frost_sources_df = get_all_frost_sources().pipe(clean_frost_sources)
# First generate a CSV using the MapFarmerToAddress script
farmer_df = pd.read_csv("data/farmers_with_address.csv")

farmers_with_sensors_df = assign_farmers_to_weather_sensor(farmer_df, frost_sources_df)
farmers_subset = farmers_with_sensors_df.head(50)

start_date = '2017-04-01'
end_date = '2017-10-01'
save_path = "drive/My Drive/Kornmo/data"

# reference time in the FrostAPI
ref_time = f'{start_date}/{end_date}'

farmers_subset_with_weather = (farmers_with_sensors_df.head(5)
                               .pipe(apply_temperature, ref_time)
                               .pipe(apply_precipitation, ref_time)
                               .pipe(save_to_file, start_date, end_date, save_path)
                               )
