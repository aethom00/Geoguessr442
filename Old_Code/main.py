from Agent1 import Agent1
import time
import os

def log_data(model_name, total_elapsed_time, total_epochs, epoch_per_epochs, images_per_epoch, loss, accuracy, folder='Logs'):
    # Log filename format: [date]_[time]_log.txt
    log_filename = f"{time.strftime('%Y-%m-%d_%H-%M')}_log.txt"
    log_path = os.path.join(folder, model_name, log_filename)

    if not os.path.exists(folder):
        os.makedirs(folder)

    if not os.path.exists(os.path.join(folder, model_name)):
        os.makedirs(os.path.join(folder, model_name))

    with open(log_path, 'w') as file:
        file.write(f"Model name: {model_name}\n")
        file.write(f"Total elapsed time: {total_elapsed_time:.2f} seconds\n")
        file.write(f"Total epochs: {total_epochs}\n")
        file.write(f"Epochs per epoch: {epoch_per_epochs}\n")
        file.write(f"Images per epoch: {images_per_epoch}\n")
        file.write(f"Total Images Generated: {total_epochs * images_per_epoch}\n")
        file.write(f"Loss: {loss}\n")
        file.write(f"Accuracy: {accuracy}\n")

model_name = 'vision_model_v1'
agent = Agent1()
agent.init(model_path=f"{model_name}.keras")

total_elapsed_time, total_epochs, epoch_per_epochs, images_per_epoch = (
agent.train_model(
    total_epochs=200,                  # Total number of epochs to train for
    epoch_per_epochs=3 ,               # Number of sub-epochs to train for [Utiziling the same images of that epoch]
    batch_size=32,                     # Batch size for training
    images_per_epoch=10,               # How many images to generate per epoch
    use_noisy_images=True,             # Whether to use noisy images or not
    save_after_epochs=10,              # Save the model after every x epochs
))

loss, accuracy = (
agent.evaluate_model(images_to_test=100, use_noisy_images=False, show=False)
)

log_data(model_name, total_elapsed_time, total_epochs, epoch_per_epochs, images_per_epoch, loss, accuracy)

