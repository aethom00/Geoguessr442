from gui import GUI
from generate_images import get_images, generate_image
import numpy as np
import os

class Agent:
    def __init__(self, rectangles=(15, 10)):
        self.app_token = 'MLY|6743037905784607|b0ff2a68e1d5bdc77f0c775a050547b8'
        self.gui = GUI(rectangles[0], rectangles[1])

        self.city_images = np.array([])

    def init(self):
        self.gui.init()

    def generate_images(self, iterations, limit=True):
        self.clear_data_folder()
        for i in range(iterations):
            print(f"Iteration {i + 1}", end=': ')
            generate_image(self.gui, self.app_token, limit)
            self.city_images = np.append(self.city_images, get_images())
            self.clear_data_folder()

    def clear_data_folder(self):
        data_folder = 'Data'
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
