from kornmo import KornmoDataset
from frost import FrostDataset
from geodata import get_farmer_elevation
from visualize import plot, generate_alternative_outcomes
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import kornmo_utils as ku
from dense_model import train_simple_dense

kornmo = KornmoDataset()
frost = FrostDataset()

if __name__ == '__main__':
    data = kornmo.get_deliveries()\
        .pipe(ku.split_farmers_on_type)

    weather_data = frost.get_as_aggregated(1)
    data = data.merge(weather_data, on=['year', 'orgnr'])

    elevation_data = get_farmer_elevation()
    data = data.merge(elevation_data, on=['orgnr'])

    historical_data = ku.get_historical_production(kornmo, data.year.unique(), 4)
    data = data.merge(historical_data, on=['orgnr', 'year'])

    data.dropna(inplace=True)

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

    y_column = ['y']
    remove_from_training = ['orgnr', 'kommunenr', 'gaardsnummer', 'bruksnummer', 'festenummer', 'year'] + y_column

    train, val = train_test_split(shuffle(data), test_size=0.2)
    val, test = train_test_split(val, test_size=0.2)
    train_x = train.drop(remove_from_training, axis=1).to_numpy()
    train_y = train[y_column].to_numpy()

    val_x = val.drop(remove_from_training, axis=1).to_numpy()
    val_y = val[y_column].to_numpy()

    model = train_simple_dense(train_x, train_y, val_x, val_y)

    area_type = "dekar"
    plot(model, val_x, val_y)
    generate_alternative_outcomes(test, model, y_column, remove_from_training, area_type)
