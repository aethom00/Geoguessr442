import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random


im = Image.open('america.png')
width, height = im.size

num_rects_width = 10
num_rects_height = 10
square_amount = (width / num_rects_width, height / num_rects_height)

extreme_points = {'N': 49.38407, 'S': 25.11567, 'E': 66.94975, 'W': 124.73004}
height_dist = extreme_points['N'] - extreme_points['S']
width_dist = extreme_points['W'] - extreme_points['E']

def pixel_loc_to_lat_long(w, h):
    longitude = round(extreme_points['E'] + (w / width) * width_dist, 2)
    latitude = round(extreme_points['S'] + (h / height) * height_dist, 2)
    return (latitude, longitude)

def long_lat_to_index(long, lat):
    i = (long - extreme_points['E']) / width_dist * width
    j = (extreme_points['N'] - lat) / height_dist * height
    return (int(i), int(j))

locations = {}

fig, ax = plt.subplots()
ax.imshow(im)

output = [[random.uniform(0, 1) for _ in range(num_rects_width)] for _ in range(num_rects_height)]

for i, w in enumerate(range(num_rects_width)):
    for j, h in enumerate(range(num_rects_height)):
        x, y = (w * (square_amount[0]), h * square_amount[1])
        ax.add_patch( patches.Rectangle( (x, y), 
                        square_amount[0], square_amount[1], 
                        fc =(1, 0, 0, output[j][i] * 0.8),  
                        ec ='k', 
                        lw = 1) ) 
        locations[(i, j)] = pixel_loc_to_lat_long(x + 0.5 * square_amount[0], y + 0.5 * square_amount[1])
        plt.text(x, y + 0.5 * square_amount[1], locations[(i, j)], fontsize = 5 * 10/num_rects_width)

plt.show()