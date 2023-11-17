import os
import math
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class CityImage:
    def __init__(self, img_location: str, shape=(100, 100)):
        self.img_loc = img_location
        img = Image.open(self.img_loc).resize(shape, Image.ANTIALIAS)
        img_grayscale = np.array(ImageOps.grayscale(img), dtype=np.float32) / 255.0  # Normalized
        self.img_grayscale = img_grayscale
        img.close()

    # Set the location of the image
    def set_loc(self, longitude: float, latitude: float):
        self.lat = latitude
        self.long = longitude

    # Get the location of the image
    def get_loc(self):
        return self.long, self.lat

    # Distance formula with context of curvature of earth
    def haversine_distance(self, pred_lat: float, pred_long: float, in_meters=False):
        R = 6371 # Radius of the Earth in km
        dLat = math.radians(pred_lat - self.lat)
        dLon = math.radians(pred_long - self.long)
        a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(self.lat)) * math.cos(math.radians(pred_lat)) * math.sin(dLon/2) * math.sin(dLon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c # Distance in km
        return distance * 1000 if in_meters else distance # Convert to meters if required

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


def get_images(amount=float('inf'), shape=(100, 100)):
    images = []
    folder_path = 'Data'

    # Check if the directory exists
    if not os.path.exists(folder_path):
        print(f"Directory '{folder_path}' not found.")
        return np.array(images)

    for i, filename in enumerate(os.listdir(folder_path)):
        if i >= amount:  # Limit the number of images processed
            break

        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Extract longitude and latitude from filename
            try:
                long, lat = map(float, filename[:-4].split(', '))
            except ValueError:
                print(f"Invalid filename format: {filename}")
                continue

            img_location = os.path.join(folder_path, filename)
            city_image = CityImage(img_location, shape)
            city_image.set_loc(long, lat)
            images.append(city_image)

    return np.array(images)
