import math
import numpy as np

from PIL import Image, ImageOps 


class CityImage:
    def __init__(self, img_location: str):
        self.img_loc = img_location

    def set_loc(self, latitude: float, longitude: float):
        self.long = longitude
        self.lat = latitude
        self.city = self.get_city(latitude, longitude)
    
    # Get distance from actual location to predicted location
    def distance_from(self, pred_lat: float, pred_long: float):
        return math.sqrt((self.lat - pred_lat) ** 2 + (self.long - pred_long) ** 2)
    
    # Get grayscale np array
    def get_black_white_np_array(self):
        img = Image.open(self.img_loc)
        img_grayscale = ImageOps.grayscale(img)

        return np.array(img_grayscale)

        
    