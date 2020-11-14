from kornmo import KornmoDataset
from frost import FrostDataset
from geodata import get_farmer_elevation
from visualize import plot

import kornmo_utils as ku
from dense_model import train_simple_dense

kornmo = KornmoDataset()
frost = FrostDataset()

if __name__ == '__main__':
    data = kornmo.get_deliveries()\
        .pipe(ku.split_farmers_on_type)

    weather_data = frost.get_as_aggregated(7)
    data = data.merge(weather_data, on=['year', 'orgnr'])

    elevation_data = get_farmer_elevation()
    data = data.merge(elevation_data, on=['orgnr'])

    historical_data = ku.get_historical_production(kornmo, data.year.unique(), 4)
    data = data.merge(historical_data, on=['orgnr', 'year'])

    data.dropna(inplace=True)

    data.drop('orgnr', axis=1, inplace=True)
    data.drop('komnr', axis=1, inplace=True)
    data.drop('year', axis=1, inplace=True)

    data['y'] = data['levert'] / data['areal']
    data.drop('levert', axis=1, inplace=True)

    data['y'] = ku.normalize(data['y'])
    data['areal'] = ku.normalize(data['areal'])
    data['fulldyrket'] = ku.normalize(data['fulldyrket'])
    data['overflatedyrket'] = ku.normalize(data['overflatedyrket'])
    data['tilskudd_dyr'] = ku.normalize(data['tilskudd_dyr'])
    data['growth_start_day'] = ku.normalize(data['growth_start_day'])
    data['lat'] = ku.normalize(data['lat'])
    data['elevation'] = ku.normalize(data['elevation'])

    (train_x, train_y, val_x, val_y) = ku.prepare_train_validation(data, 'y', 0.2)

    model = train_simple_dense(train_x, train_y, val_x, val_y)

    plot(model, val_x, val_y)
