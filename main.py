from kornmo import Kornmo
from frost import Frost

if __name__ == '__main__':

    data = Kornmo().get_deliveries()
    print("main data", data)

    weather_data = Frost().get_as_aggregated(30)
    print("weather data", weather_data)
