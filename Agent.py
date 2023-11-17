from gui import GUI
from city_image import CityImage
from generate_images import generate_image
import os

class Agent:
    def __init__(self, rectangles=(15, 10), max_images=15):
        self.app_token = 'MLY|6743037905784607|b0ff2a68e1d5bdc77f0c775a050547b8'
        self.gui = GUI(rectangles[0], rectangles[1])
        self.max_images = max_images

        self.city_images = []

    def init(self):
        self.gui.init()

    def show(self):
        self.gui.show()

    def generate_images(self):
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
