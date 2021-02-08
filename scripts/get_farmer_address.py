import requests
import pandas as pd


def get_farmers_df(path_to_delivery_csv_file):
    unique_farmers = pd.read_csv(path_to_delivery_csv_file).drop_duplicates('orgnr')
    return unique_farmers[['orgnr']]


def get_brreg_data(orgnr):
    url = f"https://data.brreg.no/enhetsregisteret/api/enheter/{orgnr}"
    request = requests.get(url)
    req_as_json = request.json()
    if request.status_code == 200:
        return req_as_json
    else:
        print('Error! Returned status code %s' % request.status_code)
        return None


def get_address_and_commune(farmers_df):
    for index, row in farmers_df.iterrows():
        orgnr = row['orgnr']

        if index % 500 == 0:
            print(f"Status: downloaded {index} of total {len(farmers_df)}. {(index / len(farmers_df)) * 100}%")

        brreg_data = get_brreg_data(orgnr)

        if brreg_data:
            try:
                commune = brreg_data['forretningsadresse']['kommune']
                commune_id = brreg_data['forretningsadresse']['kommunenummer']

                postal_code = brreg_data['forretningsadresse']['postnummer']
                postal_place = brreg_data['forretningsadresse']['poststed']
                address = brreg_data['forretningsadresse']['adresse']

                farmers_df.at[index, 'commune'] = commune
                farmers_df.at[index, 'postal_code'] = postal_code
                farmers_df.at[index, 'postal_place'] = postal_place
                farmers_df.at[index, 'address'] = address
                farmers_df.at[index, 'brreg_commune_id'] = commune_id
            except:
                print(f"Error - missing pieces of data from source. orgnr {orgnr}")

    # Remove farmers with invalid data
    cleaned_farmers_df = farmers_df.dropna(thresh=4)  # thresh = number of valid columns
    return cleaned_farmers_df


farmers_df = get_farmers_df("data/leveransedata.csv")
farmers_df_populated = get_address_and_commune(farmers_df)
farmers_populated_renamed_df = farmers_df_populated.rename(columns={'navn': 'name', 'komnr': 'commune_id'})
farmers_populated_renamed_df.to_csv("data/farmers_with_address.csv", index=False)
