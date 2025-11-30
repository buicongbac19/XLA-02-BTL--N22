"""
Module định nghĩa kiến trúc model cho combined dataset (MNIST + Shapes)
"""
from tensorflow.keras import layers, models


"""
Module định nghĩa kiến trúc model cho combined dataset (MNIST + Shapes)
"""
from tensorflow.keras import layers, models


def build_combined_model(input_shape=(28, 28, 1), num_classes=18):
    """
    Build optimized Sequential model cho combined dataset (MNIST + Shapes)
    18 classes: 0-9 (digits) + 8 shapes

    Args:
        input_shape (tuple): Shape of input image (default: (28, 28, 1))
        num_classes (int): Number of output classes (default: 18)

    Returns:
        keras.Model: Compiled Sequential model
    """

    model = models.Sequential(name='Combined_MNIST_Shapes_Optimized')

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Block 1: Basic feature extraction (32 filters)
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1'))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2'))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool1'))
    model.add(layers.Dropout(0.25, name='dropout1'))

    # Block 2: Intermediate features (64 filters)
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv3'))
    model.add(layers.BatchNormalization(name='bn3'))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv4'))
    model.add(layers.BatchNormalization(name='bn4'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(layers.Dropout(0.3, name='dropout2'))

    # Block 3: Advanced features (128 filters)
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv5'))
    model.add(layers.BatchNormalization(name='bn5'))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv6'))
    model.add(layers.BatchNormalization(name='bn6'))

    # Global pooling for spatial invariance
    model.add(layers.GlobalAveragePooling2D(name='global_pool'))

    # Dense layers
    model.add(layers.Dense(256, activation='relu', name='dense1'))
    model.add(layers.BatchNormalization(name='bn7'))
    model.add(layers.Dropout(0.4, name='dropout3'))

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))

    return model
