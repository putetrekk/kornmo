from tqdm.autonotebook import tqdm
from typing import List
import pandas as pd
import requests


def update_farm_ids(df: pd.DataFrame, lookup_table=None, columns=["kommunenr","gaardsnummer","bruksnummer","festenummer"]):
    """
    Change all occurences of a commune/farm id with the replaced ids from the given lookup_table.
    If no lookup table is supplied, we default to the table stored in 'data/geonorge/nye_gaards_og_bruksnummer.csv'.

    example:
    > farms
    	        komnr   gnr     bnr     fnr
        0	    1702	97	    3   	0
        1	    1702	97	    16  	0
        2	    1702	97	    48  	0
        3	    704	    46	    3   	0
        4	    538	    127	    2   	0
        ...	...	...	...	...
        44573	3018	117	    11	    0
        44574	3019	66	    1	    0
        44575	419 	53	    11	    0
        44576	412	    15  	14	    0
        44577	412	    143	    2	    0
    
    > update_farm_ids(farms, 'data/geonorge/nye_gaards_og_bruksnummer.csv', ["komnr", "gnr", "bnr", "fnr"])
    	        komnr   gnr     bnr     fnr
        0	    5006    97      3       0
        1	    5006    97      16      0
        2	    5006    97      48      0
        3	    3803    46      3       0
        4	    3448    127     2       0
        ...	...	...	...	...
        44573	3018    117     11      0
        44574	3019    66      1       0
        44575	3415    53      11      0
        44576	3411    15      14      0
        44577	3411    143     2       0
    """

    if lookup_table is None or isinstance(lookup_table, str):
        file = lookup_table or "data/geonorge/nye_gaards_og_bruksnummer.csv"
        lookup_table = pd.read_csv(file, engine='python', sep=None)
    
    old_columns = list(lookup_table.filter(regex=".*_old").columns)
    new_columns = list(lookup_table.filter(regex=".*_new").columns)
    
    lookup_table = df.merge(lookup_table, left_on=columns, right_on=old_columns, how='left')
    
    update = lookup_table[new_columns]
    update.columns = columns

    updated_df = df.copy(deep=True)
    updated_df.update(update)
    return updated_df


def create_farm_id_translate_table(df: pd.DataFrame, columns=["kommunenr","gaardsnummer","bruksnummer","festenummer"]):
    """
    For a dataframe with farmers, each identified by kommunenr, gårdsnummer, bruksnummer, and festenummer, 
    returns a new dataframe with new and updated ids for every farmer, along with the old ids.
    """

    old_farms = df[columns]
    
    tqdm.pandas(desc="Creating translate table...", ncols=100)

    def apply_func(farm):
        return get_updated_commune_and_farm_id(*farm)
    
    new_farms = old_farms.progress_apply(apply_func, axis=1, result_type="broadcast")
    
    new_farms.columns = list(map(lambda c: c + "_new", columns))
    old_farms.columns = list(map(lambda c: c + "_old", columns))

    return old_farms.merge(new_farms, left_index=True, right_index=True)


def get_updated_commune_and_farm_id(kommunenr, gardsnr, bruksnr, festenr='', seksjonsnr=''):
    """
    For a commune and farm (matrikkel number/id), returns the updated ids for the same farm.
    """

    url = f'https://ws.geonorge.no/kommunereform/v1/endringer/'+\
          f'{kommunenr:0>4}-{gardsnr}-{bruksnr}-{festenr:0>1}-{seksjonsnr:0>1}'
    
    result = requests.get(url)
    if result.ok:
        result = result.json()['data']
    
    if 'erstattetav' in result:
        kommunenr, farm = result['erstattetav'].split("-")
        gardsnr, bruksnr, festenr, _ = farm.split("/")
        return get_updated_commune_and_farm_id(kommunenr, gardsnr, bruksnr, festenr, seksjonsnr)
    return [kommunenr, gardsnr, bruksnr, festenr]


def get_points_of_location(kommunenr, gardsnr, bruksnr=None, festenr=None):
    """
    Get a list of geographical coordinates (lat, lng) for a given farm, taken from geonorge address search.

    Example:
    > get_location(3014, 427, 8)
    [(59.6096543, 11.1044264)]
    """

    coordSystem = 4326  # WGS84
    maxResults = 100
    url = f'https://ws.geonorge.no/adresser/v1/sok?'+\
          f'sokemodus=AND'+\
          f'&gardsnummer={gardsnr}'+\
          (f'&bruksnummer={bruksnr}' if bruksnr is not None else '')+\
          (f'&festenummer={festenr}' if festenr is not None else '')+\
          f'&kommunenummer={kommunenr}'+\
          f'&utkoordsys={coordSystem}'+\
          f'&asciiKompatibel=false'+\
          f'&treffPerSide={maxResults}'+\
          f'&side=0'
    
    places = requests.get(url).json()
    places = places['adresser']
    points = map(lambda place: (place['representasjonspunkt']['lat'], place['representasjonspunkt']['lon']), places)

    return list(points)
