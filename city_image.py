import math
import numpy as np

from PIL import Image, ImageOps 

# Based on Google
city_centers = {
    "San Fransisco": (37.7749, -122.4194),
    "Detroit": (42.3314, -83.0458),
    "Chicago": (41.8781, -87.6298),
    "Washington DC": (38.9072, -77.0369),
    "New York City": (40.7128, -74.0060)
}

class CityImage:
    def __init__(self, img_location: str):
        self.img_loc = img_location

    def set_loc(self, latitude: float, longitude: float):
        self.long = longitude
        self.lat = latitude
        self.city = self.get_city(latitude, longitude)

    # Get city based on closest distance
    def get_city(latitude: float, longitude: float):
        distances = {city: 0 for city in city_centers.keys()}

        for city in city_centers.keys():
            lat_c, long_c = city_centers[city]

            # distance formula disregarding sqr root
            distances[city] = (latitude - lat_c) ** 2 + (longitude - long_c) ** 2

        return min(distances, key=distances.get)
    
    # Get distance from actual location to predicted location
    def distance_from(self, pred_lat: float, pred_long: float):
        return math.sqrt((self.lat - pred_lat) ** 2 + (self.long - pred_long) ** 2)
    
    # Get grayscale np array
    def get_black_white_np_array(self):
        img = Image.open(self.img_loc)
        img_grayscale = ImageOps.grayscale(img)

        return np.array(img_grayscale)

        
    