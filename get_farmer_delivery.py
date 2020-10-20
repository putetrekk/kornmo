import pandas as pd
from functools import reduce


def get_deliveries(year, url):
    deliveries = pd.read_csv(url, sep=";")
    deliveries.insert(0, 'year', year)
    deliveries.orgnr = deliveries.orgnr.astype(int)

    # 2013-2015 uses a different name for bygg_saakorn_kg, let's fix that!
    return deliveries.rename(columns={"bygg_sakorn_kg": "bygg_saakorn_kg"})


per_year_deliveries = [
    get_deliveries(2018, "http://hotell.difi.no/download/ldir/leveransedata-korn/2018-2019?download"),
    get_deliveries(2017, "http://hotell.difi.no/download/ldir/leveransedata-korn/2017-2018?download"),
    get_deliveries(2016, "http://hotell.difi.no/download/ldir/leveransedata-korn/2016-2017?download"),
    get_deliveries(2015, "http://hotell.difi.no/download/ldir/leveransedata-korn/2015-2016?download"),
    get_deliveries(2014, "http://hotell.difi.no/download/ldir/leveransedata-korn/2014-2015?download"),
    get_deliveries(2013, "http://hotell.difi.no/download/ldir/leveransedata-korn/2013-2014?download"),
]

deliveries = pd.concat(per_year_deliveries, ignore_index=True)

# Create a dataframe with summed values for each crop type
sum_cols = {
    'bygg_sum':     ['bygg_for_kg',     'bygg_mat_kg',          'bygg_saakorn_kg'       ],
    'erter_sum':    ['erter_for_kg',    'erter_mat_kg',         'erter_saakorn_kg'      ],
    'havre_sum':    ['havre_for_kg',                            'havre_saakorn_kg'      ],
    'hvete_sum':    ['hvete_for_kg',    'hvete_mat_kg',         'hvete_saakorn_kg'      ],
    'rug_sum':      ['rug_for_kg',      'rug_mat_kg',           'rug_saakorn_kg'        ],
    'rughvete_sum': ['rughvete_for_kg',                         'rughvete_saakorn_kg'   ],
    'oljefro_sum':  ['oljefro_kg',                              'oljefro_saakorn_kg'    ],
}

deliveries_summed = deliveries.filter(items=['year', 'orgnr', 'komnr'])

for sum_column, columns in sum_cols.items():
    sums = reduce(lambda a, b: a.add(b), [deliveries[column] for column in columns])
    df = pd.DataFrame(sums, columns=[sum_column])
    deliveries_summed = deliveries_summed.join(df)


# Download production grant datasets and concatenate
def get_grants_data(url):
    tilskudd = pd.read_csv(
        url,
        sep=";",
        usecols=['soeknads_aar', 'orgnr', 'fulldyrket', 'overflatedyrket'],
        dtype={'soeknads_aar': int, 'orgnr': int, 'fulldyrket': int, 'overflatedyrket': int}
    )
    return tilskudd.rename(columns={'soeknads_aar': 'year'})


per_year_grants = [
    get_grants_data("http://hotell.difi.no/download/ldir/produksjon-og-avlosertilskudd/2018?download"),
    get_grants_data("http://hotell.difi.no/download/ldir/produksjon-og-avlosertilskudd/2017?download"),
    # get_grants_data("http://hotell.difi.no/download/ldir/produksjon-og-avlosertilskudd/2016?download"),
    # get_grants_data("http://hotell.difi.no/download/ldir/produksjon-og-avlosertilskudd/2015?download"),
    # get_grants_data("http://hotell.difi.no/download/ldir/produksjon-og-avlosertilskudd/2014?download"),
    # get_grants_data("http://hotell.difi.no/download/ldir/produksjon-og-avlosertilskudd/2013?download"),
]

grants = pd.concat(per_year_grants, ignore_index=True)

# Merge the two datasets above
data = deliveries_summed.merge(grants)

data = data[lambda df: (df.bygg_sum > 0) ^
                       (df.erter_sum > 0) ^
                       (df.havre_sum > 0) ^
                       (df.hvete_sum > 0) ^
                       (df.rug_sum > 0) ^
                       (df.rughvete_sum > 0) ^
                       (df.oljefro_sum > 0)]

# Eksporter tabellen som csv
data.to_csv("leveransedata.csv", index=False)
