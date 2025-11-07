# server.py

import flwr as fl
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple

# --- Merged GAN Models ---
LATENT_DIM = 100
IMG_SHAPE = (28, 28, 1)

def build_generator():
    """Defines the Generator model architecture."""
    # This uses the Keras Functional API, which is slightly more robust
    from tensorflow.keras.layers import Input, Dense, Reshape, BatchNormalization, LeakyReLU
    from tensorflow.keras.models import Model

    noise_in = Input(shape=(LATENT_DIM,))
    x = Dense(256)(noise_in)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(np.prod(IMG_SHAPE), activation='tanh')(x)
    img_out = Reshape(IMG_SHAPE)(x)
    return Model(noise_in, img_out, name="generator")

# --- Server Logic ---

def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[float, Dict[str, fl.common.Scalar]]:

    if server_round == 0 or server_round % 30 == 0 or server_round == 2500:
        print(f"\n--- Saving images for server-side evaluation round {server_round} ---")
        model = build_generator()
        model.set_weights(parameters)

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, LATENT_DIM))
        gen_imgs = model.predict(noise, verbose=0)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(10, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        
        fig.savefig(f"round_{server_round}_generated_images.png")
        plt.close()
        print(f"Saved generated images to round_{server_round}_generated_images.png")
    
    return 0.0, {}

# Define the strategy for the server
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, 
    min_available_clients=3, 
    min_fit_clients=3,
    evaluate_fn=evaluate_fn,
)

# Start the Flower server, listening on all network interfaces
print("Starting Flower server...")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=2500),
    strategy=strategy
)

print("Server finished.")
