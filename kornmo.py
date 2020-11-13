import pandas as pd
from typing import List
from kornmo_utils import flatmap


def _filter_crops(deliveries: pd.DataFrame, crops: List[str]) -> pd.DataFrame:
    """
    Removes all columns which do not belong to the crops in the supplied list
    """

    if crops is None:
        crops = ['havre', 'hvete', 'bygg', 'rug_og_rughvete']

    all_crop_cols = deliveries \
        .filter(regex='^.*_(sum|areal)$') \
        .columns

    crop_cols_to_keep = flatmap(lambda x: (f'{x}_sum', f'{x}_areal'), crops)

    cols_to_drop = set(all_crop_cols) - set(crop_cols_to_keep)

    return deliveries \
        .drop(columns=cols_to_drop)


class KornmoDataset:
    def __init__(self):
        self.deliveries: pd.DataFrame | None = None
        self.grants: pd.DataFrame | None = None
        self.legacy_grants: pd.DataFrame | None = None

    def get_deliveries(self, crops=None, exclude_høsthvete=False) -> pd.DataFrame:
        self.__load_deliveries()

        deliveries = self.deliveries.copy(deep=True)
        if exclude_høsthvete:
            deliveries = deliveries[lambda x: x['høsthvete_areal'] == 0]

        self.__load_grants()
        data: pd.DataFrame = deliveries.merge(self.grants)

        # Combine 'vårhvete' and 'høsthvete', and 'rug' and 'rughvete'
        data['hvete_areal'] = data['vårhvete_areal'] + data['høsthvete_areal']
        data['rug_og_rughvete_sum'] = data['rug_sum'] + data['rughvete_sum']

        # ... then remove the old values
        data.drop(['vårhvete_areal', 'høsthvete_areal', 'rug_sum', 'rughvete_sum'], axis=1, inplace=True)

        # Aggregate deliveries per farm per year
        data = data.groupby(by=['orgnr', 'year'], as_index=False).agg({
            'komnr': 'mean',
            'bygg_sum': 'sum',
            'erter_sum': 'sum',
            'havre_sum': 'sum',
            'hvete_sum': 'sum',
            'oljefro_sum': 'sum',
            'rug_og_rughvete_sum': 'sum',
            'fulldyrket': 'mean',
            'overflatedyrket': 'mean',
            'tilskudd_dyr': 'mean',
            'bygg_areal': 'mean',
            'havre_areal': 'mean',
            'rug_og_rughvete_areal': 'mean',
            'hvete_areal': 'mean'
        })

        return _filter_crops(data, crops)

    def get_legacy_data(self) -> pd.DataFrame:
        self.__load_deliveries()
        deliveries = self.deliveries.copy(deep=True)

        self.__load_legacy_grants()
        data = deliveries.merge(self.legacy_grants)

        return data

    def __load_deliveries(self):
        if self.deliveries is not None:
            return

        print(f'Loading deliveries...')
        try:
            self.deliveries = pd.read_csv('data/farmer_deliveries.csv')
        except FileNotFoundError:
            from get_farmer_deliveries import data as deliveries
            self.deliveries = deliveries
        print(f'Number of deliveries loaded: {len(self.deliveries)}')

    def __load_grants(self):
        if self.grants is not None:
            return

        try:
            self.grants = pd.read_csv('data/farmer_grants.csv')
        except FileNotFoundError:
            from get_farmer_grants import data as grants
            self.grants = grants

    def __load_legacy_grants(self):
        if self.legacy_grants is not None:
            return

        print(f'Loading historical grants data...')
        try:
            self.legacy_grants = pd.read_csv('data/legacy_grants.csv')
        except FileNotFoundError:
            from get_legacy_grants import data as legacy_grants
            self.legacy_grants = legacy_grants
        print(f'Historical data loaded for years {self.legacy_grants.year.min()} to {self.legacy_grants.year.max()}.')
