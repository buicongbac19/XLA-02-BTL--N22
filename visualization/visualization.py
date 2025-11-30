"""
Module chứa các hàm visualization cho training và evaluation
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

sys.path.insert(0, r'E:\PTIT\IntSys\btl_xla')


def plot_combined_samples(images, labels, num_samples=30, save_path='visualization/combined_samples.png'):
    """
    Hiển thị mẫu ảnh từ combined dataset

    Args:
        images: Array of images
        labels: Array of labels
        num_samples: Number of samples to plot (default: 30)
        save_path: Path to save the figure (default: 'combined_samples.png')
    """
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(5, 6, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        class_name = get_class_name(labels[i])
        plt.xlabel(f"{labels[i]}: {class_name}", fontsize=8)
    plt.suptitle('Combined Dataset Samples (MNIST + Shapes)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    # Ensure target directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved combined samples to '{save_path}'")
    plt.close()


def plot_training_history(history, save_path='visualization/combined_training_history.png'):
    """
    Vẽ biểu đồ training history

    Args:
        history: Training history object
        save_path: Path to save the figure (default: 'combined_training_history.png')
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'],
                 label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'],
                 label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy (Combined Dataset)',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss (Combined Dataset)',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # Ensure target directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training history to '{save_path}'")
    plt.close()


def plot_confusion_matrix(test_labels, predicted_classes, save_path='visualization/combined_confusion_matrix.png', num_classes=18, class_names=None):
    """
    Vẽ confusion matrix

    Args:
        test_labels: True labels
        predicted_classes: Predicted labels
        save_path: Path to save the figure (default: 'combined_confusion_matrix.png')
    """
    cm = confusion_matrix(test_labels, predicted_classes)

    plt.figure(figsize=(14, 12))
    if class_names is None:
        class_names = [get_class_name(i) for i in range(num_classes)]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Combined Dataset (MNIST + Shapes)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    # Ensure target directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to '{save_path}'")
    plt.close()


def plot_predictions(test_images, test_labels, predicted_classes, num_samples=30, save_path='visualization/combined_predictions.png', num_classes=18, class_names=None):
    """
    Hiển thị kết quả dự đoán

    Args:
        test_images: Array of test images
        test_labels: True labels
        predicted_classes: Predicted labels
        num_samples: Number of samples to plot (default: 30)
        save_path: Path to save the figure (default: 'combined_predictions.png')
    """
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    axes = axes.ravel()

    if class_names is None:
        class_names = [get_class_name(i) for i in range(num_classes)]

    for i in range(num_samples):
        axes[i].imshow(test_images[i].reshape(28, 28), cmap='gray')

        pred_label = predicted_classes[i]
        true_label = test_labels[i]

        pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)

        color = 'green' if pred_label == true_label else 'red'

        axes[i].set_title(f'True: {true_name}\nPred: {pred_name}',
                          color=color, fontweight='bold', fontsize=9)
        axes[i].axis('off')

    plt.suptitle('Prediction Results (Green=Correct, Red=Wrong)',
                 fontsize=16, fontweight='bold')
    # Ensure target directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved predictions to '{save_path}'")
    plt.close()
