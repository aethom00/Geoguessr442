import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import numpy as np


im = Image.open('america.png')
width, height = im.size

num_rects_width = 13
num_rects_height = 10
square_amount = (width / num_rects_width, height / num_rects_height)

extreme_points = {'N': 50, 'S': 24, 'E': -66, 'W': -126}
total_lat = extreme_points['N'] - extreme_points['S']
total_long = extreme_points['W'] - extreme_points['E']

def pixel_loc_to_lat_long(w, h):
    return (round(extreme_points['N'] - h/height * (total_lat), 2), round(extreme_points['W'] - w/width * (total_long), 2))

def long_lat_to_index(long, lat):
    # TODO
    # Convert Coordinates to the cloest 2d index that can be used in the locations dictionary
    return ()

locations = {}

x_ticks = [i * square_amount[0] for i in range(num_rects_width + 1)]
y_ticks = [j * square_amount[1] for j in range(num_rects_height + 1)]
x_labels = [pixel_loc_to_lat_long(tick, 0)[1] for tick in x_ticks]
y_labels = [pixel_loc_to_lat_long(0, tick)[0] for tick in y_ticks]

fig, ax = plt.subplots()
ax.imshow(im)

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

ax.set_xticklabels(x_labels, rotation=90)
ax.set_yticklabels(y_labels)

output = np.array([[random.uniform(0, 1) for _ in range(num_rects_width)] for _ in range(num_rects_height)])

for i, w in enumerate(range(num_rects_width)):
    for j, h in enumerate(range(num_rects_height)):
        x, y = (w * square_amount[0], h * square_amount[1])
        ax.add_patch(patches.Rectangle((x, y), 
                                      square_amount[0], square_amount[1], 
                                      fc=(1, 0, 0, output[j, i] * 0.5),  
                                      ec='none', 
                                      lw=1))
        locations[(i, j)] = pixel_loc_to_lat_long(x + 0.5 * square_amount[0], y + 0.5 * square_amount[1])
        plt.text(x, y + 0.5 * square_amount[1], locations[(i, j)], fontsize=5 * 10/num_rects_width)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()  
plt.show()
