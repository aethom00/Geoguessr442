import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import numpy as np

class GUI: # class for image
    def __init__(self, num_rects_width, num_rects_height):
        self.image_loc = 'scaled_america_mex_can.png' # I think this might be a better representation
        im = Image.open(self.image_loc) # opens image
        self.width, self.height = im.size
        im.close()

        # assigns number of x,y rectangles to input numbers
        self.num_rects_width = num_rects_width 
        self.num_rects_height = num_rects_height

        self.square_amount = (self.width / num_rects_width, self.height / num_rects_height)
        self.extreme_points = {'N': 50, 'S': 24, 'E': -66, 'W': -126} # assigns extreme points
        self.total_lat = self.extreme_points['N'] - self.extreme_points['S']
        self.total_long = self.extreme_points['W'] - self.extreme_points['E']
        self.output = np.zeros((self.num_rects_height, self.num_rects_width))
        self.show_ticks = True
        self.locations = {}
        self.dots = []

    def pixel_loc_to_lat_long(self, w, h):
        return (round(self.extreme_points['N'] - h/self.height * (self.total_lat), 2), round(self.extreme_points['W'] - w/self.width * (self.total_long), 2))

    def long_lat_to_pixel(self, long, lat):
        # Calculation for longitude and lat relative position
        long_rel = (self.extreme_points['W'] - long) / self.total_long
        lat_rel = (self.extreme_points['N'] - lat) / self.total_lat

        x_pixel = int(long_rel * self.width)
        y_pixel = int(lat_rel * self.height)

        return (x_pixel, y_pixel)

    
    def init(self): 
        self.x_ticks = [i * self.square_amount[0] for i in range(self.num_rects_width + 1)]
        self.y_ticks = [j * self.square_amount[1] for j in range(self.num_rects_height + 1)]
        self.x_labels = [self.pixel_loc_to_lat_long(tick, 0)[1] for tick in self.x_ticks]
        self.y_labels = [self.pixel_loc_to_lat_long(0, tick)[0] for tick in self.y_ticks]

        for i, w in enumerate(range(self.num_rects_width)): # i represents an incremented variable starting at 0 and increasing by 1
            for j, h in enumerate(range(self.num_rects_height)): # j represents an incremented variable starting at 0 and increasing by 1
                x, y = (w * self.square_amount[0], h * self.square_amount[1])
                self.locations[(i, j)] = self.pixel_loc_to_lat_long(x + 0.5 * self.square_amount[0], y + 0.5 * self.square_amount[1])

    def place_dot(self, long, lat, color=None, r=10):
        x_pixel, y_pixel = self.long_lat_to_pixel(long, lat)
        circle = plt.Circle((x_pixel, y_pixel), r, color=(color if color else 'blue'), fill=True)
        self.dots.append(circle) # store the dot object

    def clear_dots(self):
        for dot in self.dots:
            dot.remove()  # Remove the dot from the plot
        self.dots.clear()  # Clear the list of stored dots

    def toggle_ticks(self, show_ticks):
        self.show_ticks = show_ticks
    
    def show(self, display_coords=False): 
        _, map = plt.subplots()
        map.imshow(Image.open(self.image_loc))

        if self.show_ticks:
            map.set_xticks(self.x_ticks)
            map.set_yticks(self.y_ticks)

            map.set_xticklabels(self.x_labels, rotation=90)
            map.set_yticklabels(self.y_labels)
        else:
            map.axis('off')

        map.set_xlabel("Longitude (West)")
        map.set_ylabel("Latitude (North)")

        for i, w in enumerate(range(self.num_rects_width)):
            for j, h in enumerate(range(self.num_rects_height)):
                x, y = (w * self.square_amount[0], h * self.square_amount[1])
                map.add_patch(patches.Rectangle(
                                    (x, y), 
                                    self.square_amount[0], 
                                    self.square_amount[1], 
                                    fc=(1, 0, 0, self.output[j, i] * 0.5),  
                                    ec='none', 
                                    lw=1))
                if display_coords:
                    map.text(x, y + 0.5 * self.square_amount[1], self.locations[(i, j)], fontsize=5 * 10/self.num_rects_width)
        
        for dot in self.dots:
            map.add_patch(dot)

        plt.tight_layout()
        plt.show()


    def generate_random_output(self):
        self.output = np.array([[random.uniform(0, 1) for _ in range(self.num_rects_width)] for _ in range(self.num_rects_height)])

    def clear_output(self):
        self.output.fill(0)

    def map_output_to_grid(self, model_output):
            reshaped_output = model_output.reshape(self.num_rects_height, self.num_rects_width)

            normalized_output = reshaped_output - reshaped_output.min()
            if normalized_output.max() != 0:
                normalized_output /= normalized_output.max()

            self.output = normalized_output

