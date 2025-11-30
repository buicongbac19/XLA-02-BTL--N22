"""
Module chứa các hàm evaluation và đánh giá model
"""

import sys
import numpy as np
from sklearn.metrics import classification_report


def evaluate_model(model, test_data, num_classes=18, class_names=None):
    """
    Đánh giá model trên test set

    Args:
        model: Trained keras model
        test_data: tuple (test_images, test_labels)

    Returns:
        ndarray: Predicted classes
    """
    test_images, test_labels = test_data

    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)

    # Evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Predictions
    predictions = model.predict(test_images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)

    if class_names is None:
        class_names = [get_class_name(i) for i in range(num_classes)]

    # Ensure labels argument matches the expected classes (avoids mismatch when some classes missing)
    labels = list(range(num_classes))
    print(classification_report(test_labels,
          predicted_classes, labels=labels, target_names=class_names))

    # Per-category accuracy
    print("\n" + "="*70)
    print("PER-CATEGORY ACCURACY")
    print("="*70)

    for i in range(num_classes):
        mask = test_labels == i
        if mask.sum() > 0:
            acc = (predicted_classes[mask] == test_labels[mask]).mean()
            class_name = class_names[i] if i < len(class_names) else get_class_name(i)
            print(
                f"Class {i:2d} ({class_name:15s}): {acc*100:.2f}% ({mask.sum()} images)")

    print("="*70)

    return predicted_classes
