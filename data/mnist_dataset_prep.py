import tensorflow as tf
import numpy as np

def load_and_partition_data(num_clients=3):
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape the data
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)

    # Create a partition for each client
    partition_size = len(x_train) // num_clients
    partitions = []
    for i in range(num_clients):
        start = i * partition_size
        end = start + partition_size
        partitions.append((x_train[start:end], y_train[start:end]))

    return partitions, (x_test, y_test)

client_partitions, test_set = load_and_partition_data(num_clients=3)
# Save each client's partition to a separate file
for i, client_data in enumerate(client_partitions):
    file_path = f"client_{i+1}_data.npz"
    np.savez(file_path, x=client_data[0], y=client_data[1])
    print(f"Saved {file_path} with {len(client_data[0])} samples.")

# Save the global test set
test_file_path = "test_data.npz"
np.savez(test_file_path, x=test_set[0], y=test_set[1])
print(f"Saved {test_file_path} with {len(test_set[0])} samples.")
