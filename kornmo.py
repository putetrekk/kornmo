import pandas as pd
from typing import List
from kornmo_utils import flatmap


class KornmoDataset:
    def __init__(self):
        self.deliveries: pd.DataFrame | None = None

    def get_deliveries(self, crops=None, exclude_høsthvete=False) -> pd.DataFrame:
        if self.deliveries is not None:
            return self._filter_crops(crops).copy(deep=True)

        print(f'Loading deliveries...')

        try:
            deliveries = pd.read_csv('data/leveransedata.csv')
        except FileNotFoundError:
            import get_farmer_delivery
            deliveries = pd.read_csv('data/leveransedata.csv')

        if exclude_høsthvete:
           deliveries = deliveries[lambda x: x['høsthvete_areal'] == 0]

        # Combine 'vårhvete' and 'høsthvete', and 'rug' and 'rughvete'
        deliveries['hvete_areal'] = deliveries['vårhvete_areal'] + deliveries['høsthvete_areal']
        deliveries['rug_og_rughvete_sum'] = deliveries['rug_sum'] + deliveries['rughvete_sum']

        # ... then remove the old values
        deliveries.drop(['vårhvete_areal', 'høsthvete_areal', 'rug_sum', 'rughvete_sum'], axis=1, inplace=True)

        # Remove farmers which have multiple entries for the same season.
        # TODO: Consider merging entries instead
        duplicate_filter = deliveries.duplicated(subset=['year', 'orgnr'], keep=False)
        deliveries = deliveries[~duplicate_filter]
        deliveries.reset_index(drop=True, inplace=True)

        self.deliveries = deliveries
        print(f'Number of deliveries loaded: {len(self.deliveries)}')
        return self._filter_crops(crops).copy(deep=True)

    def _filter_crops(self, crops: List[str]) -> pd.DataFrame:
        """
        Removes all columns which do not belong to the crops in the supplied list
        """

        if crops is None:
            crops = ['havre', 'hvete', 'bygg', 'rug_og_rughvete']

        all_crop_cols = self.deliveries\
            .filter(regex='^.*_(sum|areal)$')\
            .columns

        crop_cols_to_keep = flatmap(lambda x: (f'{x}_sum', f'{x}_areal'), crops)

        cols_to_drop = set(all_crop_cols) - set(crop_cols_to_keep)

        return self.deliveries\
            .drop(columns=cols_to_drop)
