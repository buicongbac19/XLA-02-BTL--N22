"""
Module để load Shape dataset
"""
import os
import sys

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Add path before importing local modules


from preprocessing.image_processing import (  # noqa: E402
    rgb_to_grayscale,
    apply_threshold_inverted,
    find_bounding_box,
    crop_image,
    add_padding_square,
    resize_image,
    add_border_padding,
    normalize_image
)


def load_shape_data(shape_dataset_path, label_offset=0):
    """
    Load shape dataset (labels 0..7 by default).

    Shape mapping (default):
    - circle: 0
    - kite: 1
    - parallelogram: 2
    - rectangle: 3
    - rhombus: 4
    - square: 5
    - trapezoid: 6
    - triangle: 7

    Args:
        shape_dataset_path (str): Path to shape dataset folder
        label_offset (int): Offset added to class indices (default 0)

    Returns:
        tuple: ((train_images, train_labels), (val_images, val_labels), (test_images, test_labels))
    """
    print("\n" + "="*70)
    print("LOADING SHAPE DATASET")
    print("="*70)

    shape_classes = ['circle', 'kite', 'parallelogram', 'rectangle',
                     'rhombus', 'square', 'trapezoid', 'triangle']

    # Class mapping (applies label_offset so callers can choose numbering)
    class_to_label = {shape: idx + label_offset for idx, shape in enumerate(shape_classes)}

    print(f"Shape dataset path: {shape_dataset_path}")
    print(f"Shape classes: {shape_classes}")
    print(f"Label mapping: {class_to_label}")

    def load_images_from_folder(folder_path, label_offset=label_offset, use_preprocessing=True):
        """
        Helper function to load images from folder structure
        Áp dụng các hàm preprocessing từ image_processing.py
        """
        images = []
        labels = []
        failed_images = []

        for class_idx, class_name in enumerate(shape_classes):
            class_folder = os.path.join(folder_path, class_name)
            if not os.path.exists(class_folder):
                print(f"Warning: Folder not found: {class_folder}")
                continue

            image_files = [f for f in os.listdir(class_folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            print(f"  {class_name}: {len(image_files)} images", end='')

            success_count = 0
            for img_file in image_files:
                img_path = os.path.join(class_folder, img_file)
                try:
                    if use_preprocessing:
                        # Áp dụng pipeline xử lý ảnh đầy đủ
                        # Load ảnh gốc
                        pil_image = Image.open(img_path)
                        original = np.array(pil_image)

                        # Xử lý ảnh qua pipeline
                        try:
                            # 1. Grayscale
                            gray = rgb_to_grayscale(original)

                            # 2. Threshold & Invert (nền đen, hình trắng)
                            thresh = apply_threshold_inverted(gray)

                            # 3. Crop theo bounding box
                            x_min, y_min, x_max, y_max = find_bounding_box(
                                thresh)
                            cropped = crop_image(
                                thresh, x_min, y_min, x_max, y_max)

                            # 4. Padding vuông
                            padded_square = add_padding_square(cropped)

                            # 5. Resize về 20x20
                            resized_20 = resize_image(padded_square, 20)

                            # 6. Thêm border padding 4px → 28x28
                            final_28 = add_border_padding(
                                resized_20, padding=4)

                            # 7. Normalize về [0, 1]
                            normalized = normalize_image(final_28)

                            # Reshape để match định dạng (28, 28, 1)
                            img_array = normalized.reshape(28, 28, 1)

                            images.append(img_array)
                            labels.append(class_idx + label_offset)
                            success_count += 1

                        except ValueError as ve:
                            # Nếu không tìm thấy bounding box, dùng phương pháp đơn giản
                            img = load_img(
                                img_path, color_mode='grayscale', target_size=(28, 28))
                            img_array = img_to_array(img) / 255.0
                            images.append(img_array)
                            labels.append(class_idx + label_offset)
                            success_count += 1
                            failed_images.append(
                                (img_file, class_name, str(ve)))
                    else:
                        # Phương pháp đơn giản: Load và resize trực tiếp
                        img = load_img(
                            img_path, color_mode='grayscale', target_size=(28, 28))
                        img_array = img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(class_idx + label_offset)
                        success_count += 1

                except Exception as e:
                    failed_images.append((img_file, class_name, str(e)))
                    continue

            print(f" → {success_count} processed successfully")

        if failed_images:
            print(
                f"\n  ⚠ Warning: {len(failed_images)} images failed preprocessing (used simple method)")
            if len(failed_images) <= 5:
                for img_file, class_name, error in failed_images:
                    print(f"    - {class_name}/{img_file}: {error}")

        return np.array(images), np.array(labels)

    # Load train, val, test với preprocessing
    print("\nLoading training images with preprocessing...")
    train_images, train_labels = load_images_from_folder(
        os.path.join(shape_dataset_path, 'train'),
        label_offset=label_offset,
        use_preprocessing=True)

    print("\nLoading validation images with preprocessing...")
    val_images, val_labels = load_images_from_folder(
        os.path.join(shape_dataset_path, 'val'),
        label_offset=label_offset,
        use_preprocessing=True)

    print("\nLoading test images with preprocessing...")
    test_images, test_labels = load_images_from_folder(
        os.path.join(shape_dataset_path, 'test'),
        label_offset=label_offset,
        use_preprocessing=True)

    print(f"\nShape dataset loaded:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    print("="*70)

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
