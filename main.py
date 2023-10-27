import numpy as np
import tensorflow as tf
import pandas as pd
import scipy as sp
import os
from os import listdir
from city_image import CityImage

images = []
for image in os.listdir("Images"):
    images.append(CityImage(image))

# coords = pd.read_csv("coords.csv")
# for i, coord in enumerate(coords):
#     images[i].set_loc(coord[0], coord[1])


