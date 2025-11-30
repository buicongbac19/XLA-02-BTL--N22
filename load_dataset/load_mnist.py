"""
Module để load MNIST dataset
"""
import numpy as np
from tensorflow.keras.datasets import mnist


def load_mnist_data():
    """
    Load MNIST dataset
    Labels: 0-9 (giữ nguyên)

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    print("="*70)
    print("LOADING MNIST DATASET")
    print("="*70)

    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    print(f"Training set: {train_images.shape[0]} images")
    print(f"Test set: {test_images.shape[0]} images")

    # Reshape và normalize
    train_images = train_images.reshape(-1,
                                        28, 28, 1).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    print(f"Train images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")
    print("="*70)

    return train_images, train_labels, test_images, test_labels
