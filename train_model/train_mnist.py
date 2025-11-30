

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
# Ensure project root is on sys.path so sibling packages import correctly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from load_dataset.load_mnist import load_mnist_data
from train_model.model_architecture import build_combined_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from train_model.evaluation import evaluate_model as shared_evaluate_model
from visualization.visualization import (
    plot_training_history as shared_plot_training_history,
    plot_confusion_matrix as shared_plot_confusion_matrix,
    plot_predictions as shared_plot_predictions,
)
import os
from datetime import datetime


def build_model(input_shape=(28, 28, 1), num_classes=10):
    """Use the project's shared model architecture for MNIST (10 classes)."""
    return build_combined_model(input_shape=input_shape, num_classes=num_classes)


def load_and_preprocess_data():
    """Load MNIST dataset using the shared loader and split validation set."""
    mnist_train_imgs, mnist_train_labels, mnist_test_imgs, mnist_test_labels = load_mnist_data()

    # Split validation set same as original
    val_split = 10000
    val_images = mnist_train_imgs[-val_split:]
    val_labels = mnist_train_labels[-val_split:]
    train_images = mnist_train_imgs[:-val_split]
    train_labels = mnist_train_labels[:-val_split]

    test_images = mnist_test_imgs
    test_labels = mnist_test_labels

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def plot_sample_images(images, labels, num_samples=25):
    """
    Hiển thị mẫu ảnh từ dataset
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.xlabel(f"Label: {labels[i]}")
    plt.suptitle('./Sample MNIST Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    # Ensure visualization directory exists
    save_dir = os.path.join('.', 'visualization')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'sample_images.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved sample images to '{save_path}'")
    plt.close()


def train_model(model, train_data, val_data, epochs=30, batch_size=128):
    """
    Train model với callbacks
    """
    train_images, train_labels = train_data
    val_images, val_labels = val_data

    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    print("\nModel Architecture:")
    model.summary()

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'mnist_{timestamp}.h5'

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Data Augmentation
    print("\n" + "="*70)
    print("DATA AUGMENTATION CONFIGURATION")
    print("="*70)
    datagen = ImageDataGenerator(
        rotation_range=30,           # Xoay ±30 độ
        width_shift_range=0.1,       # Dịch ngang 10%
        height_shift_range=0.1,      # Dịch dọc 10%
        zoom_range=0.1,              # Zoom ±10%
        fill_mode='nearest'          # Điền pixel khi transform
    )

    print("✓ Rotation range: ±10°")
    print("✓ Width/Height shift: ±10%")
    print("✓ Zoom range: ±10%")
    print("✓ Shear range: 0.1")
    print("="*70)

    # Fit augmentation on training data
    datagen.fit(train_images)

    # Train
    print(f"\nStarting training with data augmentation...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Model will be saved to: {model_path}\n")

    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=batch_size),
        steps_per_epoch=len(train_images) // batch_size,
        epochs=epochs,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)

    return history, model_path



def main():
    """
    Main training pipeline
    """
    print("\n" + "="*70)
    print("MNIST TRAINING PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # 1. Load data
    (train_images, train_labels), (val_images,
                                   val_labels), (test_images, test_labels) = load_and_preprocess_data()

    # 2. Plot sample images
    plot_sample_images(train_images, train_labels)

    # 3. Build model
    print("\n" + "="*70)
    print("BUILDING  MODEL")
    print("="*70)
    model = build_model()
    print("✓ Model built successfully!")

    # 4. Train model
    history, model_path = train_model(
        model,
        (train_images, train_labels),
        (val_images, val_labels),
        epochs=30,
        batch_size=128
    )

    # 5. Plot training history (shared utility)
    shared_plot_training_history(history, save_path='./visualization/mnist_training_history.png')

    # 6. Evaluate model (shared utility) - MNIST has 10 classes
    predicted_classes = shared_evaluate_model(model, (test_images, test_labels), num_classes=10)

    # 7. Plot confusion matrix (shared utility)
    shared_plot_confusion_matrix(test_labels, predicted_classes, save_path='./visualization/mnist_confusion_matrix.png', num_classes=10)

    # 8. Plot predictions (shared utility)
    shared_plot_predictions(test_images, test_labels, predicted_classes, save_path='./visualization/mnist_predictions.png', num_samples=20, num_classes=10)

    # 9. Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Training history plot: training_history.png")
    print(f"✓ Confusion matrix: confusion_matrix.png")
    print(f"✓ Predictions plot: predictions.png")
    print(f"✓ Sample images: sample_images.png")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    return model, history


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run training
    model, history = main()

    print("\nTraining completed successfully!")
