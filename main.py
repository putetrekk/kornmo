from kornmo import KornmoDataset
from frost import FrostDataset

from utils import split_farmers_on_type, prepare_train_validation, normalize
from dense_model import train_simple_dense

kornmo = KornmoDataset()
frost = FrostDataset()

if __name__ == '__main__':
    deliveries = kornmo.get_deliveries()\
        .pipe(split_farmers_on_type)
    weather_data = frost.get_as_aggregated(7)
    data = deliveries.merge(weather_data)

    data.drop('orgnr', axis=1, inplace=True)
    data.drop('komnr', axis=1, inplace=True)
    data.drop('year', axis=1, inplace=True)

    data['y'] = normalize(data['levert'] / data['areal'])
    data.drop('levert', axis=1, inplace=True)

    data_split = prepare_train_validation(data, 'y', 0.1)

    train_simple_dense(*data_split)

