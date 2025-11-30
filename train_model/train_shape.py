

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train_model.evaluation import evaluate_model as shared_evaluate_model
from visualization.visualization import (
    plot_training_history as shared_plot_training_history,
    plot_confusion_matrix as shared_plot_confusion_matrix,
    plot_predictions as shared_plot_predictions,
)
from train_model.model_architecture import build_combined_model
from load_dataset.load_shapes import load_shape_data


def build_model(input_shape=(28, 28, 1), num_classes=8):
    """Use shared model builder for shape classification (8 classes)."""
    return build_combined_model(input_shape=input_shape, num_classes=num_classes)


def load_and_preprocess_data():
    """Load shape dataset using the shared loader in `load_dataset/load_shapes.py`.

    Expects a folder `shape_dataset` as sibling to `train_model` (same as before).
    Returns: ((train_images, train_labels), (val_images, val_labels), (test_images, test_labels), class_names)
    """
    # Relative path from train_model folder to shape_dataset folder
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'shape_dataset')
    data_dir = os.path.abspath(data_dir)

    print("="*70)
    print("LOADING SHAPE DATASET (via load_shape_data)")
    print("="*70)
    print(f"Shape data directory: {data_dir}")

    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = load_shape_data(data_dir, label_offset=0)

    # Build class names in same order as loader
    class_names = ['circle', 'kite', 'parallelogram', 'rectangle',
                   'rhombus', 'square', 'trapezoid', 'triangle']

    return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels), class_names


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
    plt.suptitle('Sample Shape Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./visualization/shape_sample_images.png', dpi=150, bbox_inches='tight')
    print("✓ Saved sample images to './visualization/shape_sample_images.png'")
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
    model_path = f'shape_classifier_{timestamp}.h5'

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
        rotation_range=10,           # Xoay ±10 độ
        width_shift_range=0.1,       # Dịch ngang 10%
        height_shift_range=0.1,      # Dịch dọc 10%
        zoom_range=0.1,              # Zoom ±10%
        shear_range=0.1,             # Shear transformation
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


# Use shared evaluation/visualization utilities (shape-only trainer)


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*70)
    print("SHAPE CLASSIFICATION TRAINING PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # 1. Load data
    (train_images, train_labels), (val_images,
                                   val_labels), (test_images, test_labels), class_names = load_and_preprocess_data()

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
    shared_plot_training_history(history, save_path='./visualization/shape_training_history.png')

    # 6. Evaluate model (shared utility) - shapes have 8 classes
    predicted_classes = shared_evaluate_model(model, (test_images, test_labels), num_classes=8, class_names=class_names)

    # 7. Plot confusion matrix (shared utility)
    shared_plot_confusion_matrix(test_labels, predicted_classes, save_path='./visualization/shape_confusion_matrix.png', num_classes=8, class_names=class_names)

    # 8. Plot predictions (shared utility)
    shared_plot_predictions(test_images, test_labels, predicted_classes, num_samples=20, save_path='./visualization/shape_predictions.png', num_classes=8, class_names=class_names)

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
