import math

# Initial code gotten from https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y
# with some minor adjustments
class GeoPointTranslator:
    # Initiate with the center coordinates of the farm.
    # A bounding box will be created, to represent the frame
    # of the sentinel image.
    radius = 6371  # Earth Radius in Km

    def __init__(self, bounding_box_geo):
        self.lng_min, self.lat_min, self.lng_max, self.lat_max = bounding_box_geo.exterior.bounds
        self.p0 = {
            'srcX': 0,
            'srcY': 0,
            'lat': self.lat_min,
            'lng': self.lng_min
        }

        self.p1 = {
            'srcX': 100,
            'srcY': 100,
            'lat': self.lat_max,
            'lng': self.lng_max
        }

        # Calculate global X and Y for top-left reference point
        self.p0['pos'] = self.lat_lng_to_global_xy(self.p0['lat'], self.p0['lng'])
        # Calculate global X and Y for bottom-right reference point
        self.p1['pos'] = self.lat_lng_to_global_xy(self.p1['lat'], self.p1['lng'])

    # Converts lat and lng coordinates to SCREEN X and Y positions
    def lat_lng_to_global_xy(self, lat, lng):
        x = self.radius * lng * math.cos((self.p0['lat'] + self.p1['lat']) / 2)
        y = self.radius * lat

        return {'x': x, 'y': y}

    def lat_lng_to_screen_xy(self, lat, lng):
        # Calculate global X and Y for projection point
        pos = self.lat_lng_to_global_xy(lat, lng)
        # Calculate the percentage of Global X position in relation to total global width
        pos['perX'] = ((pos['x'] - self.p0['pos']['x']) / (self.p1['pos']['x'] - self.p0['pos']['x']))
        # Calculate the percentage of Global Y position in relation to total global height
        pos['perY'] = ((pos['y'] - self.p0['pos']['y']) / (self.p1['pos']['y'] - self.p0['pos']['y']))

        # Returns the screen position based on reference points
        return {
            'x': self.p0['srcX'] + (self.p1['srcX'] - self.p0['srcX']) * pos['perX'],
            'y': self.p0['srcY'] + (self.p1['srcY'] - self.p0['srcY']) * pos['perY']
        }
