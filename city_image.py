import os
import math
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class CityImage:
    def __init__(self, img_location: str, shape=(100, 100)):
        self.img_loc = img_location
        img = Image.open(self.img_loc).resize(shape)
        self.img_grayscale = np.array(ImageOps.grayscale(img), dtype=np.float32) / 255.0  # Normalized
        self.grid_x, self.grid_y = None, None
        img.close()

    # Set the location of the image
    def set_loc(self, longitude: float, latitude: float):
        self.lat = latitude
        self.long = longitude

    # Get the location of the image
    def get_loc(self):
        return self.long, self.lat

    # old
    # # Distance formula with context of curvature of earth
    # def haversine_distance(self, pred_lat: float, pred_long: float, in_meters=False):
    #     R = 6371 # Radius of the Earth in km
    #     dLat = math.radians(pred_lat - self.lat)
    #     dLon = math.radians(pred_long - self.long)
    #     a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(self.lat)) * math.cos(math.radians(pred_lat)) * math.sin(dLon/2) * math.sin(dLon/2)
    #     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    #     distance = R * c # Distance in km
    #     return distance * 1000 if in_meters else distance # Convert to meters if required
    

    # Get the image
    def get_image(self, noisy=False):
        return self.img_grayscale if not noisy else self.get_image_with_noise()
    
    # Get Image with added noise
    def get_image_with_noise(self, mean=0, var=0.01):
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, self.img_grayscale.shape)
        noisy_img = np.clip(self.img_grayscale + gaussian, 0, 1)  # Ensuring it's still between 0 and 1
        return noisy_img

    # Display image
    def show(self, noisy=False):
        plt.imshow(self.get_image(noisy=noisy), cmap='gray')
        plt.title(f"Location: {self.get_loc()}")
        plt.xticks([])  
        plt.yticks([])  
        plt.show()

    def set_grid_pos(self, gui):
        pixel_x, pixel_y = gui.long_lat_to_pixel(self.long, self.lat)

        self.grid_x = int(pixel_x // gui.square_amount[0])
        self.grid_y = int(pixel_y // gui.square_amount[1])


# def haversine_distance_david(predicted_lat, predicted_long, actual_lat, actual_long, in_meters=False): # predicted and true should be tuples
#     R = 6371 # Radius of the Earth in km

#     dLat = math.radians(predicted_lat - actual_lat)
#     dLon = math.radians(predicted_long - actual_long)
#     a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(actual_lat)) * math.cos(math.radians(predicted_lat)) * math.sin(dLon/2) * math.sin(dLon/2)
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
#     distance = R * c # Distance in km
#     return distance * 1000 if in_meters else distance # Convert to meters if required

# def haversine_distance_ashton(predicted_lat, predicted_long, actual_lat, actual_long, in_meters=False): # predicted and true should be tuples
#     R = 6371 # Radius of the Earth in km

#     predicted_lat = math.radians(predicted_lat)
#     predicted_long = math.radians(predicted_long)
#     actual_lat = math.radians(actual_lat)
#     actual_long = math.radians(actual_long)

#     delta_phi = actual_lat - predicted_lat
#     delta_lambda = actual_long - predicted_long
#     distance = 2 * R * math.asin(np.sqrt((math.sin(delta_phi/2))**2 + (math.cos(predicted_lat) * math.cos(actual_lat) * (np.sin(delta_lambda/2))**2)))
#     return distance * 1000 if in_meters else distance # Convert to meters if required

# def haversine_distance_claire(predicted_lat, predicted_long, actual_lat, actual_long, in_meters=False): # predicted and true should be tuples
#     # distance haversine 
#     R = 6371 # Radius of the Earth in km
#     rad = 2 * R 
#     # convert lat and longitudes to radians 
#     predicted_lat, predicted_long  = math.radians(predicted_lat), math.radians(predicted_long)
#     actual_lat, actual_long  = math.radians(actual_lat), math.radians(actual_long)
#     dlat = 1 - math.cos(actual_lat - predicted_lat)
#     dlon = math.cos(predicted_lat) * math.cos(actual_lat) * (1 - math.cos(actual_long - predicted_long))
#     distance = rad * math.asin(math.sqrt((dlat + dlon)/2))
#     return distance * 1000 if in_meters else distance # Convert to meters if required

def haversine_distance(predicted_lat, predicted_long, actual_lat, actual_long, in_meters=False): # predicted and true should be tuples
    # distance haversine 
    R = 6371 # Radius of the Earth in km
    rad = 2 * R 
    # convert lat and longitudes to radians 
    predicted_lat, predicted_long  = math.radians(predicted_lat), math.radians(predicted_long)
    actual_lat, actual_long  = math.radians(actual_lat), math.radians(actual_long)
    dlat = 1 - math.cos(actual_lat - predicted_lat)
    dlon = math.cos(predicted_lat) * math.cos(actual_lat) * (1 - math.cos(actual_long - predicted_long))
    distance = rad * math.asin(math.sqrt((dlat + dlon)/2))
    return distance * 1000 if in_meters else distance # Convert to meters if required
