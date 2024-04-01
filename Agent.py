from gui import GUI
from generate_images import get_images, generate_images
import numpy as np
import os
import time
import tensorflow

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model

# added
import shutil

class Agent:
    def __init__(self, rectangles=(15, 10)):
        self.app_token = 'MLY|7380996212029520|cd6ad220b67aff81fe1328c76de2255a'
        self.gui = GUI(rectangles[0], rectangles[1])
        self.city_images = np.array([])
        

    def setup_network(self, model_path):
        self.output_size = self.gui.num_rects_height * self.gui.num_rects_width
        self.input_shape = (100, 100, 1)
        self.model = create_cnn_model(self.input_shape, self.output_size, model_path=model_path)

    def init(self, model_path='vision_model.keras'): # default constructor
        self.gui.init()
        self.setup_network(model_path)
        self.model_path = model_path

    def generate_images(self, iterations, limit=1, verbose=False, image_cache=set()):
        self.clear_data_folder() # will wipe all previous data
        for i in range(iterations):
            generate_images(self.gui, self.app_token, limit, verbose=verbose, image_cache=image_cache)
            self.city_images = np.append(self.city_images, get_images())
            self.clear_data_folder()
            print(f"Generated {i + 1}/{iterations} image(s)", end='\r')

        for city_image in self.city_images:
            city_image.set_grid_pos(self.gui)
        
        print(f"Done generating {len(self.city_images)} image(s)")

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

    def refresh(self, amount, verbose=False, image_cache=set()):
        self.city_images = np.array([])
        self.generate_images(amount, verbose=verbose, image_cache=image_cache)

    def predict_images_in_folder(self, folder_path='Images', use_noisy_images=False, show=True):
        loaded_images = get_images(shape=self.input_shape[:-1], folder_path=folder_path)

        for city_image in loaded_images:
            img_data = city_image.get_image(noisy=use_noisy_images).reshape(1, *self.input_shape)
            prediction = self.model.predict(img_data)[0]

            if show:
                self.gui.map_output_to_grid(prediction)
                self.gui.clear_dots()
                # Unkown location corresponds to coordinates (0, 0)
                if city_image.get_loc() != (0, 0):
                    self.gui.place_dot(*city_image.get_loc(), color='blue') 
                city_image.show() 
                self.gui.show()  

    def train_model(self, total_epochs=10, epoch_per_epochs=10, batch_size=32, images_per_epoch=5, use_noisy_images=False, save_after_epochs=10):
        total_start_time = time.time()
        image_cache = set()

        for epoch in range(total_epochs):
            epoch_start_time = time.time()
            print(f"Epoch {epoch + 1}/{total_epochs} ...")
            self.refresh(images_per_epoch, verbose=False, image_cache=image_cache)

            # Prepare the dataset for the current epoch
            X_train = np.array([ci.get_image(noisy=use_noisy_images).reshape(100, 100, 1) for ci in self.city_images])
            y_train = np.array([self.gui.num_rects_width * ci.grid_y + ci.grid_x for ci in self.city_images])
            y_train = np.eye(self.output_size)[y_train]

            self.model.fit(X_train, y_train, epochs=epoch_per_epochs, batch_size=batch_size)

            epoch_end_time = time.time()
            elapsed_time = epoch_end_time - epoch_start_time
            estimated_total_time = elapsed_time * (total_epochs - epoch - 1)
            print(f"Elapsed time for epoch: {elapsed_time:.2f} seconds")
            print(f"Estimated time remaining: {estimated_total_time:.2f} seconds")

            if (epoch + 1) % save_after_epochs == 0:
                print("Saving model...")
                self.model.save(self.model_path)

        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        print(f"Total training time: {total_elapsed_time:.2f} seconds")
        self.model.save(self.model_path)

        return total_elapsed_time, total_epochs, epoch_per_epochs, images_per_epoch

    def evaluate_model(self, images_to_test=10, use_noisy_images=False, show=True):
        print("Generating images to test...")
        self.refresh(images_to_test, verbose=False)
        print("Done generating images to test") 

        # Prepare the dataset
        X_test = np.array([ci.get_image(noisy=use_noisy_images).reshape(100, 100, 1) for ci in self.city_images])
        y_test = np.array([self.gui.num_rects_width * ci.grid_y + ci.grid_x for ci in self.city_images])

        # One-hot encoding for output
        y_test = np.eye(self.output_size)[y_test]

        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")

        # Predict and display for each test image
        if show:
            for i, city_image in enumerate(self.city_images):
                prediction = self.model.predict(X_test[i:i+1])[0]
                self.gui.map_output_to_grid(prediction)
                self.gui.clear_dots()
                self.gui.place_dot(*city_image.get_loc(), color='blue')
                city_image.show(noisy=use_noisy_images)
                self.gui.show()

        return loss, accuracy


def create_cnn_model(input_shape, num_classes, model_path):
    if os.path.exists(model_path):
        print("Loading existing model...")
        return load_model(model_path)
    else:
        print("Creating new model...")
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])
        
        model.save(model_path)
        print(f"Model saved to {model_path}")

        return model

        
#sushrita
def main():
    agent = Agent(rectangles=(15, 10))
    agent.init()

    # output_dir = "Data"

    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir, ignore_errors=True)
    # os.makedirs(output_dir)  

    iterations = 5
    limit = 1
    verbose = True
    image_cache = set()
    agent.generate_images(iterations, limit=limit, verbose=verbose, image_cache=image_cache)


    agent.show()

if __name__ == "__main__":
    main()    

    
