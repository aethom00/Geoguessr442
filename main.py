from Agent import Agent

agent = Agent()
agent.init(model_path='vision_model_v1.keras')

agent.train_model(
    total_epochs=150,                  # Total number of epochs to train for
    epoch_per_epochs=10,               # Number of sub-epochs to train for [Utiziling the same images of that epoch]
    batch_size=32,                     # Batch size for training
    images_per_epoch=20,               # How many images to generate per epoch
    use_noisy_images=False,            # Whether to use noisy images or not
    save_after_epochs=10,              # Save the model after every x epochs
)
agent.evaluate_model()
