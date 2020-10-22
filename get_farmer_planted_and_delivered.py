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

grain_production_codes = {
    'bygg': ['p242'],
    'havre': ['p243'],
    'hvete': ['p240', 'p247'],
    'rug_og_rughvete': ['p238']
}


def get_farmers_by_type_yield(year, grain_type):
    grain_codes = grain_production_codes[grain_type]

    production_grants_url = f'http://hotell.difi.no/download/ldir/produksjon-og-avlosertilskudd/{year}?download.csv'
    production_grants_df = pd.read_csv(
        production_grants_url,
        sep=";",
        dtype={'soeknads_aar': int, 'orgnr': int, 'fulldyrket': int}
    )

    relevant_columns = ['soeknads_aar', 'orgnr', 'fulldyrket'] + grain_codes
    production_grants_df = production_grants_df[relevant_columns]
    production_grants_df[grain_codes] = production_grants_df[grain_codes].fillna(0.0).astype(int)

    # Incase of multiple columns: høsthvete/vårhvete. Do we want this in the same year?
    production_grants_df[f'{grain_type}_dyrket'] = production_grants_df[grain_codes].sum(axis=1)
    production_grants_df = production_grants_df.drop(columns=grain_codes)
    return production_grants_df


grain_planted_by_year_dfs = [
    [get_farmers_by_type_yield(2017, grain_type) for grain_type in grain_production_codes.keys()],
    [get_farmers_by_type_yield(2018, grain_type) for grain_type in grain_production_codes.keys()],
    [get_farmers_by_type_yield(2019, grain_type) for grain_type in grain_production_codes.keys()]
]


def merge_grains_by_year(grains_planted_dfs):
    merged_by_year_dfs = reduce(lambda df_left, df_right: pd.merge(
        df_left,
        df_right,
        on=['soeknads_aar', 'orgnr', 'fulldyrket']
    ), grains_planted_dfs)

    merged_by_year_dfs = merged_by_year_dfs.rename(columns={"soeknads_aar": "year"})

    return merged_by_year_dfs


grain_planted_by_year_df = [merge_grains_by_year(by_year_dfs) for by_year_dfs in grain_planted_by_year_dfs]
grains_planted_df = pd.concat(grain_planted_by_year_df, ignore_index=True)

planted_and_delivered_grains = pd.merge(grains_planted_df, deliveries_summed, on=['year', 'orgnr'])

planted_and_delivered_grains.to_csv("data/mengde_dyrket_og_levert.csv", index=False)
