# XLA-N22 - MNIST & Geometric Shapes Classifier

Dự án nhận diện chữ số MNIST và hình học bằng Deep Learning với giao diện GUI và tính năng Air Writing (vẽ bằng tay trong không khí).

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Huấn luyện Model](#huấn-luyện-model)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cấu trúc Model](#cấu-trúc-model)

## 🎯 Tổng quan

Dự án này là một hệ thống nhận diện đa lớp (multi-class classification) có khả năng:

- Nhận diện chữ số từ 0-9 (MNIST dataset)
- Nhận diện 8 loại hình học: Circle, Kite, Parallelogram, Rectangle, Rhombus, Square, Trapezoid, Triangle
- Giao diện GUI thân thiện với Tkinter
- Tính năng Air Writing: vẽ bằng ngón tay trỏ qua camera với MediaPipe

## ✨ Tính năng

### 1. Nhận diện chữ số MNIST

- Upload ảnh từ file
- Dự đoán với độ tin cậy cao
- Hiển thị top 3 dự đoán

### 2. Nhận diện hình học

- 8 loại hình học được hỗ trợ
- Tên hiển thị bằng tiếng Việt
- Tương thích với cùng pipeline xử lý ảnh

### 3. Air Writing

- Vẽ bằng ngón tay trỏ qua webcam
- Smoothing filter để vẽ mượt mà
- Nhận diện real-time
- Điều khiển bằng bàn phím:
  - `d`: Bật/tắt chế độ vẽ
  - `c`: Xóa canvas
  - `p`: Dự đoán hình vẽ
  - `q`: Thoát

### 4. Xử lý ảnh thông minh

- Pipeline preprocessing tự động:
  - Grayscale conversion
  - Otsu thresholding
  - Bounding box detection
  - Square padding
  - Resize về 28x28
  - Normalization

## 📁 Cấu trúc dự án

```
btl_xla/
│
├── README.md                          # File này
├── requirements.txt                   # Danh sách dependencies
├── .gitignore                         # Git ignore rules
│
├── air_writing_core.py                # Core logic cho Air Writing
│   └── Classes: AirWritingCore, SmoothingFilter
│
├── gui_prediction.py                  # GUI application chính
│   └── Class: PredictionGUI
│
├── preprocessing/                     # Module xử lý ảnh
│   └── image_processing.py
│       ├── rgb_to_grayscale()
│       ├── otsu_threshold()
│       ├── apply_threshold_inverted()
│       ├── find_bounding_box()
│       ├── crop_image()
│       ├── add_padding_square()
│       ├── resize_image()
│       ├── add_border_padding()
│       ├── normalize_image()
│       └── Helper functions cho Air Writing
│
├── load_dataset/                      # Module load dữ liệu
│   ├── load_mnist.py
│   │   └── load_mnist_data()         # Load MNIST từ Keras
│   │
│   └── load_shapes.py
│       └── load_shape_data()         # Load shape dataset với preprocessing
│
├── train_model/                       # Module huấn luyện
│   ├── model_architecture.py
│   │   └── build_combined_model()    # Kiến trúc CNN
│   │
│   ├── train_mnist.py                # Script train MNIST model
│   │   ├── build_model()
│   │   ├── load_and_preprocess_data()
│   │   ├── train_model()
│   │   └── main()
│   │
│   ├── train_shape.py                # Script train Shape model
│   │   ├── build_model()
│   │   ├── load_and_preprocess_data()
│   │   ├── train_model()
│   │   └── main()
│   │
│   └── evaluation.py                 # Module đánh giá model
│       └── evaluate_model()          # Test accuracy, classification report
│
├── visualization/                     # Module visualization
│   └── visualization.py
│       ├── plot_training_history()   # Vẽ biểu đồ training
│       ├── plot_confusion_matrix()    # Confusion matrix
│       └── plot_predictions()        # Hiển thị kết quả dự đoán
│
├── model/                             # Thư mục chứa model đã train
│   ├── mnist_20251125_164544.h5      # MNIST model
│   └── shape_classifier_20251125_182743.h5  # Shape model
│
└── shape_dataset/                     # Dataset hình học
    ├── train/                         # Training set (1500 ảnh/class)
    │   ├── circle/
    │   ├── kite/
    │   ├── parallelogram/
    │   ├── rectangle/
    │   ├── rhombus/
    │   ├── square/
    │   ├── trapezoid/
    │   └── triangle/
    │
    ├── val/                           # Validation set (500 ảnh/class)
    │   └── [các thư mục tương tự train]
    │
    └── test/                          # Test set (500 ảnh/class)
        └── [các thư mục tương tự train]
```

### Chi tiết các module

#### 1. `air_writing_core.py`

Module xử lý logic chính cho tính năng Air Writing:

- **SmoothingFilter**: Lọc smoothing thích ứng dựa trên tốc độ di chuyển
  - Adaptive smoothing: di chuyển nhanh → ít smoothing, di chuyển chậm → nhiều smoothing
  - Exponential smoothing với hệ số alpha động
- **AirWritingCore**: Class quản lý Air Writing
  - Quản lý camera và MediaPipe Hands
  - Xử lý frame, vẽ canvas
  - Interpolation cho đường vẽ mượt mà

#### 2. `gui_prediction.py`

Giao diện người dùng chính:

- **PredictionGUI**: Class chính quản lý GUI
  - Tab "Upload": Upload ảnh và dự đoán
  - Tab "Air Writing": Vẽ bằng camera
  - Model selector: Chuyển đổi giữa MNIST và Shape model
  - Hiển thị preprocessing steps
  - Top 3 predictions

#### 3. `preprocessing/image_processing.py`

Pipeline xử lý ảnh đầy đủ:

1. **Grayscale conversion**: Chuyển RGB → Grayscale
2. **Otsu thresholding**: Tự động tìm ngưỡng tối ưu
3. **Threshold & Invert**: Đảo ngược (nền đen, chữ/hình trắng)
4. **Bounding box**: Tìm vùng chứa đối tượng
5. **Crop**: Cắt theo bounding box
6. **Square padding**: Thêm padding để thành hình vuông
7. **Resize**: Resize về 20x20
8. **Border padding**: Thêm border 4px → 28x28
9. **Normalization**: Chuẩn hóa về [0, 1]

#### 4. `load_dataset/`

- **load_mnist.py**: Load MNIST dataset từ Keras (60,000 train, 10,000 test)
- **load_shapes.py**: Load shape dataset từ thư mục, áp dụng preprocessing pipeline

#### 5. `train_model/`

- **model_architecture.py**: Định nghĩa kiến trúc CNN
  - Conv2D layers với BatchNormalization
  - MaxPooling và Dropout
  - GlobalAveragePooling
  - Dense layers
- **train_mnist.py**: Script train model MNIST (10 classes)
- **train_shape.py**: Script train model Shape (8 classes)
- **evaluation.py**: Đánh giá model với accuracy, classification report

#### 6. `visualization/`

- **visualization.py**: Các hàm vẽ biểu đồ
  - Training history (accuracy, loss)
  - Confusion matrix
  - Prediction samples

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.7+
- Webcam (cho tính năng Air Writing)
- Windows/Linux/macOS

### Cài đặt dependencies

```bash
# Clone repository
git clone <repository-url>
cd btl_xla

# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Cài đặt packages
pip install -r requirements.txt
```

### Cấu trúc thư mục cần thiết

Đảm bảo các thư mục sau tồn tại:

- `model/`: Chứa file model `.h5`
- `shape_dataset/`: Chứa dataset hình học (train/val/test)
- `visualization/`: Thư mục lưu các biểu đồ (tự động tạo)

## 💻 Sử dụng

### Chạy GUI Application

```bash
python gui_prediction.py
```

### Sử dụng GUI

1. **Chọn Model**: Dropdown ở trên cùng để chọn "Digit (MNIST)" hoặc "Shape"
2. **Tab Upload**:
   - Click "Choose Image File" để upload ảnh
   - Xem kết quả dự đoán và top 3
   - Click "Hiển thị quá trình xử lý" để xem preprocessing steps
3. **Tab Air Writing**:
   - Click "Bắt đầu Air Writing" để mở camera
   - Nhấn `d` để bật/tắt vẽ
   - Vẽ bằng ngón tay trỏ
   - Nhấn `p` để dự đoán
   - Nhấn `c` để xóa canvas
   - Nhấn `q` để thoát

## 🎓 Huấn luyện Model

### Train MNIST Model

```bash
cd train_model
python train_mnist.py
```

Model sẽ được lưu vào thư mục `model/` với tên `mnist_YYYYMMDD_HHMMSS.h5`

### Train Shape Model

```bash
cd train_model
python train_shape.py
```

Model sẽ được lưu vào thư mục `model/` với tên `shape_classifier_YYYYMMDD_HHMMSS.h5`

### Cấu hình Training

Các tham số có thể điều chỉnh trong file training:

- `epochs`: Số epoch (mặc định: 30)
- `batch_size`: Batch size (mặc định: 128)
- `patience`: Early stopping patience (mặc định: 10)
- Data augmentation: Rotation, shift, zoom, shear

### Kết quả Training

Sau khi train, các file sau sẽ được tạo trong `visualization/`:

- `mnist_training_history.png` / `shape_training_history.png`: Biểu đồ training
- `mnist_confusion_matrix.png` / `shape_confusion_matrix.png`: Confusion matrix
- `mnist_predictions.png` / `shape_predictions.png`: Mẫu dự đoán
- `sample_images.png`: Mẫu ảnh từ dataset

## 🛠️ Công nghệ sử dụng

- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV, MediaPipe
- **GUI**: Tkinter
- **Image Processing**: PIL/Pillow, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: scikit-learn
- **Data Processing**: NumPy, Pandas (nếu cần)

## 🏗️ Cấu trúc Model

### Kiến trúc CNN

```
Input (28, 28, 1)
    ↓
Conv2D(32, 5x5) + BatchNorm + ReLU
    ↓
Conv2D(32, 3x3) + BatchNorm + ReLU
    ↓
MaxPooling2D(2x2) + Dropout(0.25)
    ↓
Conv2D(64, 3x3) + BatchNorm + ReLU
    ↓
Conv2D(64, 3x3) + BatchNorm + ReLU
    ↓
MaxPooling2D(2x2) + Dropout(0.3)
    ↓
Conv2D(128, 3x3) + BatchNorm + ReLU
    ↓
Conv2D(128, 3x3) + BatchNorm + ReLU
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) + BatchNorm + ReLU + Dropout(0.4)
    ↓
Dense(num_classes) + Softmax
    ↓
Output
```

### Hyperparameters

- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**:
  - EarlyStopping (patience=10)
  - ModelCheckpoint (save best only)
  - ReduceLROnPlateau (factor=0.5, patience=5)

## 📝 Ghi chú

- Đảm bảo camera được kết nối trước khi sử dụng Air Writing
- Để có kết quả tốt nhất, vẽ trong điều kiện ánh sáng đủ

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.
