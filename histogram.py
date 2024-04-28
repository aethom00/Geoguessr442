import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    latitude = []
    longitude = []
    names = os.listdir('CombinedFiles')
    for name in names:
        name = name.removesuffix(".jpg")
        name = name.split('_')
        latitude.append(float(name[0]))
        longitude.append(float(name[1]))
    
    
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    

    plt.hist2d(longitude, latitude, bins=44, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('2D Histogram of Longitudes and Latitudes')
    plt.savefig('histogram.png')

    pass


if __name__ == '__main__':
    main()