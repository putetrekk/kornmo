import pandas as pd


def get_grants_data(year):
    columns = ["soeknads_aar", "orgnr", "kommunenr", "arealtilskudd", "husdyrtilskudd"]

    columns_to_rename = {
        "arealtilskudd": "areal_tilskudd",
        "kommunenr": "komnr",
        "orgnavn": "navn",
        "soeknads_aar": "year",
        "husdyrtilskudd": "husdyr_tilskudd"
    }

    url = f'http://hotell.difi.no/download/ldir/produksjon-og-avlosertilskudd/{year}?download'
    tilskudd = pd.read_csv(
        url,
        sep=";",
        usecols=columns
    )

    tilskudd = tilskudd.rename(columns=columns_to_rename)
    tilskudd = tilskudd.dropna().astype('int32')
    return tilskudd


def get_legacy_grants_data(year):
    columns = ["orgnr", "kommune", "t_areal", 't_husdyr']
    url = f'http://hotell.difi.no/download/ldir/produksjonstilskudd/{year}aug?download'
    tilskudd = pd.read_csv(
        url,
        sep=";",
        error_bad_lines=False,
        usecols=columns
    )

    tilskudd = tilskudd.rename(columns={"t_areal": "areal_tilskudd", "kommune": "komnr", 't_husdyr': 'husdyr_tilskudd'})
    tilskudd['year'] = year
    tilskudd = tilskudd.dropna().astype('int32')
    return tilskudd


# http://hotell.difi.no/download/ldir/produksjonstilskudd/2013aug?download
legacy_grants = [
    get_grants_data(2019),
    get_grants_data(2018),
    get_grants_data(2017),
    get_legacy_grants_data(2016),
    get_legacy_grants_data(2015),
    get_legacy_grants_data(2014),
    get_legacy_grants_data(2013)
]

data = pd.concat(legacy_grants, ignore_index=True)

# Export to csv
data.to_csv("data/legacy_grants.csv", index=False)
