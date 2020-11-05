import pandas as pd


class Kornmo:
    def __init__(self):
        self.deliveries: pd.DataFrame | None = None

    def get_deliveries(self):
        if self.deliveries is not None:
            return self.deliveries

        try:
            deliveries = pd.read_csv('data/leveransedata.csv')
        except FileNotFoundError:
            import get_farmer_delivery
            deliveries = pd.read_csv('data/leveransedata.csv')

        deliveries['hvete_areal'] = deliveries['vårhvete_areal'] + deliveries['høsthvete_areal']
        deliveries['rug_og_rughvete_sum'] = deliveries['rug_sum'] + deliveries['rughvete_sum']

        # Remove farmers which have multiple entries for the same season.
        # TODO: Consider merging entries instead
        duplicate_filter = deliveries.duplicated(subset=['year', 'orgnr'], keep=False)
        deliveries = deliveries[~duplicate_filter]

        self.deliveries = deliveries
        return self.deliveries

