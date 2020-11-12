import pandas as pd

# Download production grant datasets and concatenate
column_mapper = {
    'p242': 'bygg_areal',
    'p243': 'havre_areal',
    'p247': 'høsthvete_areal',
    'p240': 'vårhvete_areal',
    'p238': 'rug_og_rughvete_areal',
}

animal_grants = ['husdyrtilskudd', 'melkeproduksjon', 'storfekjoettproduksjon',  'utmarksbeitetilskudd']
column_2017_animal_grants_mapper = {
    'driftstilskudd_melkeproduksjon': 'melkeproduksjon',
    'driftstilskudd_ spesialisert_storfekjottproduksjon': 'storfekjoettproduksjon'
}


def get_grants_data(url):
    columns = ['soeknads_aar', 'orgnr', 'fulldyrket', 'overflatedyrket', 'tilskudd_dyr'] + list(column_mapper.keys())

    tilskudd = pd.read_csv(
        url,
        sep=";"
    )
    print(tilskudd)
    tilskudd = tilskudd.rename(columns=column_2017_animal_grants_mapper)
    tilskudd['tilskudd_dyr'] = tilskudd[animal_grants].sum(axis=1)
    tilskudd = tilskudd[columns]
    tilskudd = tilskudd.rename(columns=column_mapper)

    # convert all columns to int
    tilskudd = tilskudd.fillna(0).astype('int32')

    return tilskudd.rename(columns={'soeknads_aar': 'year'})


per_year_grants = [
    # get_grants_data("http://hotell.difi.no/download/ldir/produksjon-og-avlosertilskudd/2019?download"),
    get_grants_data("http://hotell.difi.no/download/ldir/produksjon-og-avlosertilskudd/2018?download"),
    get_grants_data("http://hotell.difi.no/download/ldir/produksjon-og-avlosertilskudd/2017?download"),
]

data = pd.concat(per_year_grants, ignore_index=True)

# Export to csv
data.to_csv("data/farmer_grants.csv", index=False)