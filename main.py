from Agent import Agent

agent = Agent()
agent.init(model_path='vision_model_v1.keras')

agent.train_model(
    total_epochs=1_000,                # Total number of epochs to train for
    epoch_per_epochs=15,               # Number of sub-epochs to train for [Utiziling the same images of that epoch]
    batch_size=32,                     # Batch size for training
    images_per_epoch=100,              # How many images to generate per epoch
    use_noisy_images=False,            # Whether to use noisy images or not
)
agent.evaluate_model()
