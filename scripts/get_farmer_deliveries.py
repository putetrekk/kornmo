import pandas as pd
from functools import reduce


def get_deliveries(year, url):
    deliveries = pd.read_csv(url, sep=";")
    deliveries.insert(0, 'year', year)
    deliveries.orgnr = deliveries.orgnr.astype(int)

    # 2013-2015 / 2019 uses a different name for saakorn.
    standardized_grain_names = {
        "bygg_sakorn_kg": "bygg_saakorn_kg",
        "havre_sakorn_kg": "havre_saakorn_kg",
        "erter_sakorn_kg": "erter_saakorn_kg",
        "hvete_sakorn_kg": "hvete_saakorn_kg",
        "rug_sakorn_kg": "rug_saakorn_kg",
        "rughvete_sakorn_kg": "rughvete_saakorn_kg",
        "oljefro_sakorn_kg": "oljefro_saakorn_kg"
    }
    return deliveries.rename(columns=standardized_grain_names)


per_year_deliveries = [
    get_deliveries(2019, "http://hotell.difi.no/download/ldir/leveransedata-korn/2019-2020?download"),
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

data = deliveries_summed

# Export to csv
data.to_csv("data/landbruksdir/raw/farmer_deliveries.csv", index=False)
