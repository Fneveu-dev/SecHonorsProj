!pip install -q flwr

import flwr as fl
import tensorflow as tf
from google.colab import drive
import numpy as np
import os
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

CLIENT_ID = 1

drive.mount('/content/drive', force_remount=True)

DRIVE_DATA_PATH = '/content/drive/Shareddrives/FedSynDrive/Data'
data_path = os.path.join(DRIVE_DATA_PATH, f'client_{CLIENT_ID}_data.npz')

print(f"--- This notebook is now CLIENT #{CLIENT_ID} ---")
print(f"Loading data from: {data_path}")

with np.load(data_path) as data:
    x_train = data['x'].astype(np.float32)

print(f"Successfully loaded {len(x_train)} samples for this client.")


LATENT_DIM = 100
IMG_SHAPE = (28, 28, 1)

def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=LATENT_DIM))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(IMG_SHAPE), activation='tanh'))
    model.add(Reshape(IMG_SHAPE))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=IMG_SHAPE))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


  
class GanClient(fl.client.NumPyClient):
    def __init__(self, x_train):
        # Rescale data to [-1, 1] for GAN training
        if x_train.max() > 1.0:
            self.x_train = (x_train - 127.5) / 127.5
        else:
            self.x_train = x_train * 2.0 - 1.0
        # Ensure data has the channel dimension
        if self.x_train.ndim == 3:
            self.x_train = np.expand_dims(self.x_train, axis=3)

        self.optimizer = Adam(0.0002, 0.5)
        self.discriminator = build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.generator = build_generator()
        z = Input(shape=(LATENT_DIM,)); img = self.generator(z)
        self.discriminator.trainable = False; validity = self.discriminator(img)
        self.combined = Model(z, validity); self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def get_parameters(self, config):
        return self.generator.get_weights()

    def fit(self, parameters, config):
        self.generator.set_weights(parameters); batch_size = 32
        valid = np.ones((batch_size, 1))
        # Train for one epoch
        for i in range(0, len(self.x_train), batch_size):
            real_imgs = self.x_train[i:i+batch_size]
            if len(real_imgs) == 0: continue
            noise = np.random.normal(0, 1, (len(real_imgs), LATENT_DIM)); gen_imgs = self.generator.predict(noise, verbose=0)
            d_loss_real = self.discriminator.train_on_batch(real_imgs, np.ones((len(real_imgs), 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((len(gen_imgs), 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
            g_loss = self.combined.train_on_batch(noise, valid)
        return self.generator.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.x_train), {}

print(f"\n--- Starting Client #{CLIENT_ID} ---")


# Start the Flower client
fl.client.start_numpy_client(
    server_address="0.tcp.ngrok.io:12217",
    client=GanClient(x_train=x_train)
)

print(f"\n--- Client #{CLIENT_ID} has finished. ---")
