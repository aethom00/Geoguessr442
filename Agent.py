from gui import GUI
from generate_images import get_images, generate_image
import numpy as np
import os

class Agent:
    def __init__(self, rectangles=(15, 10)):
        self.app_token = 'MLY|6743037905784607|b0ff2a68e1d5bdc77f0c775a050547b8'
        self.gui = GUI(rectangles[0], rectangles[1])

        self.city_images = np.array([])

    def setup_network(self):
        self.output_size = self.gui.num_rects_height * self.gui.num_rects_width

    def init(self): # default constructor
        self.gui.init()

    def generate_images(self, iterations, limit=1):
        self.clear_data_folder() # will wipe all previous data
        for i in range(iterations):
            print(f"Iteration {i + 1}", end=': ')
            generate_image(self.gui, self.app_token, limit)
            self.city_images = np.append(self.city_images, get_images())
            self.clear_data_folder()
        print("Done generating images")

    def clear_data_folder(self):
        data_folder = 'Data'
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def show(self, see_locations=True, display_coordinates=False):
        self.gui.clear_dots()
        if see_locations:
            for city_image in self.city_images:
                self.gui.place_dot(*city_image.get_loc(), color='blue')
        self.gui.show(display_coords=display_coordinates)

    

    
