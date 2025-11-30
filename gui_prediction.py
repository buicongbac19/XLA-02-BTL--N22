"""
GUI application để dự đoán MNIST digits và geometric shapes
Có 3 chức năng:
1. Upload ảnh từ file
2. Vẽ bằng chuột trên canvas
3. Air Writing với camera
"""
import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import cv2
import threading
import time

# Add path
sys.path.insert(0, r'D:\IntSys\XLA\btl_xla')

from preprocessing.image_processing import (  # noqa: E402
    rgb_to_grayscale,
    apply_threshold_inverted,
    find_bounding_box,
    crop_image,
    add_padding_square,
    resize_image,
    add_border_padding,
    normalize_image,
    bgr_to_rgb
)
from air_writing_core import AirWritingCore  # noqa: E402

try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow not available. Model loading will be disabled.")


class PredictionGUI:
    def __init__(self, root, model_path=None):
        self.root = root
        self.root.title("MNIST + Shapes Prediction")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)

        # Models - load both
        self.digit_model = None
        self.shape_model = None
        self.model = None  # Current active model
        # Mode: 'mnist' or 'shapes'
        self.mode = 'mnist'

        # Class names
        self.class_names = {i: str(i) for i in range(10)}
        self.shape_names = ['Circle', 'Kite', 'Parallelogram', 'Rectangle', 'Rhombus', 'Square', 'Trapezoid', 'Triangle']
        # Mapping tên hình học sang tiếng Việt
        self.shape_names_vn = {
            'Circle': 'Hình tròn',
            'Kite': 'Diều',
            'Parallelogram': 'Hình bình hành',
            'Rectangle': 'Hình chữ nhật',
            'Rhombus': 'Hình thoi',
            'Square': 'Hình vuông',
            'Trapezoid': 'Hình thang',
            'Triangle': 'Hình tam giác'
        }
        
        # Air Writing
        self.air_writing_core = AirWritingCore()
        
        # Store last processed image for visualization
        self.last_uploaded_image = None
        self.last_air_writing_canvas = None

        self.setup_ui()
        self.load_all_models()
    
    def get_display_name(self, class_name):
        """Chuyển đổi tên lớp sang tiếng Việt nếu là hình học"""
        if self.mode == 'shapes' and class_name in self.shape_names_vn:
            return self.shape_names_vn[class_name]
        return class_name

    def setup_ui(self):
        """Setup giao diện"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=70)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="🎨 MNIST + Shapes Predictor",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)

        # Main container with notebook (tabs)
        main_frame = tk.Frame(self.root, bg='#ffffff')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))

        # Model selector - Centered in white area
        selector_container = tk.Frame(main_frame, bg='#ffffff')
        selector_container.pack(fill=tk.X, pady=(0, 15))

        # Center container for selector
        center_frame = tk.Frame(selector_container, bg='#ffffff')
        center_frame.pack(expand=True)

        # Label
        selector_label = tk.Label(
            center_frame,
            text="Model:",
            font=('Arial', 13, 'bold'),
            bg='#ffffff',
            fg='#2c3e50'
        )
        selector_label.pack(side=tk.LEFT, padx=(0, 12))

        # Dropdown selector
        self.mode_var = tk.StringVar(value='Digit (MNIST)')
        
        # Style the combobox
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.TCombobox',
                       fieldbackground='#ffffff',
                       background='#ffffff',
                       borderwidth=2,
                       relief='solid',
                       padding=10)
        
        self.model_combobox = ttk.Combobox(
            center_frame,
            textvariable=self.mode_var,
            values=['🔢 Digit (MNIST)', '🔷 Shape'],
            state='readonly',
            font=('Arial', 12, 'bold'),
            width=20,
            style='Custom.TCombobox'
        )
        self.model_combobox.pack(side=tk.LEFT, padx=(0, 15))
        self.model_combobox.bind('<<ComboboxSelected>>', self.on_mode_change)

        # Status indicator
        self.model_status_label = tk.Label(
            center_frame,
            text="",
            font=('Arial', 11),
            bg='#ffffff',
            fg='#27ae60'
        )
        self.model_status_label.pack(side=tk.LEFT)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        # Tab 1: Upload
        self.upload_draw_frame = tk.Frame(self.notebook, bg='#ecf0f1')
        self.notebook.add(self.upload_draw_frame, text="📁 Upload")

        # Tab 2: Air Writing
        self.air_writing_frame = tk.Frame(self.notebook, bg='#ecf0f1')
        self.notebook.add(self.air_writing_frame, text="✋ Air Writing")

        # Setup tabs
        self.setup_upload_draw_tab()
        self.setup_air_writing_tab()

        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def setup_upload_draw_tab(self):
        """Setup tab Upload"""
        # Main container - centered
        main_container = tk.Frame(self.upload_draw_frame, bg='#ecf0f1')
        main_container.pack(fill=tk.BOTH, expand=True, padx=50, pady=30)

        # Left panel - Upload
        left_frame = tk.LabelFrame(
            main_container,
            text="📁 Upload Image",
            font=('Arial', 14, 'bold'),
            bg='#ecf0f1',
            padx=15,
            pady=15
        )
        left_frame.grid(row=0, column=0, padx=(0, 20), sticky='nsew')

        # Right panel - Results
        right_frame = tk.LabelFrame(
            main_container,
            text="🎯 Prediction Results",
            font=('Arial', 14, 'bold'),
            bg='#ecf0f1',
            padx=15,
            pady=15
        )
        right_frame.grid(row=0, column=1, sticky='nsew')

        # Upload button
        upload_btn = tk.Button(
            left_frame,
            text="📤 Choose Image File",
            command=self.upload_image,
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            padx=30,
            pady=15,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3
        )
        upload_btn.pack(pady=20)

        # Preview uploaded image
        self.upload_preview_label = tk.Label(
            left_frame,
            text="No image selected",
            bg='#ecf0f1',
            font=('Arial', 10, 'italic'),
            fg='#7f8c8d'
        )
        self.upload_preview_label.pack(pady=(0, 10))

        self.upload_image_label = tk.Label(
            left_frame,
            bg='#ffffff',
            relief=tk.SUNKEN,
            bd=2
        )
        self.upload_image_label.pack(pady=10, padx=20)

        # Results section
        result_frame = tk.Frame(right_frame, bg='#ecf0f1')
        result_frame.pack(fill=tk.BOTH, expand=True)

        # Processed image preview
        tk.Label(
            result_frame,
            text="Processed Image (28x28):",
            font=('Arial', 11, 'bold'),
            bg='#ecf0f1'
        ).pack(pady=(5, 0))

        self.processed_image_label = tk.Label(result_frame, bg='#ecf0f1')
        self.processed_image_label.pack(pady=5)

        # Prediction result
        self.result_label = tk.Label(
            result_frame,
            text="Upload an image\nto get prediction",
            font=('Arial', 14),
            bg='#ecf0f1',
            fg='#7f8c8d',
            justify=tk.CENTER
        )
        self.result_label.pack(pady=10)

        # Show preprocessing steps button
        self.show_steps_btn = tk.Button(
            result_frame,
            text="🔬 Hiển thị quá trình xử lý",
            command=self.show_preprocessing_steps_upload,
            font=('Arial', 11, 'bold'),
            bg='#9b59b6',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.show_steps_btn.pack(pady=(10, 5))

        # Top 3 predictions
        self.top3_frame = tk.Frame(result_frame, bg='#ecf0f1')
        self.top3_frame.pack(pady=10)

        # Configure grid weights
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(0, weight=1)

    def load_all_models(self):
        """Load both models from model directory"""
        if not KERAS_AVAILABLE:
            self.model_status_label.config(text="❌ TensorFlow not available", fg='#e74c3c')
            messagebox.showerror("Error", "TensorFlow is not installed!")
            return
        
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        
        # Load digit model
        digit_model_path = os.path.join(model_dir, 'mnist_20251125_164544.h5')
        if os.path.exists(digit_model_path):
            try:
                self.digit_model = keras.models.load_model(digit_model_path)
                print(f"✅ Loaded digit model: {digit_model_path}")
            except Exception as e:
                print(f"❌ Error loading digit model: {e}")
                self.digit_model = None
        else:
            print(f"⚠️ Digit model not found: {digit_model_path}")
            self.digit_model = None
        
        # Load shape model
        shape_model_path = os.path.join(model_dir, 'shape_classifier_20251125_182743.h5')
        if os.path.exists(shape_model_path):
            try:
                self.shape_model = keras.models.load_model(shape_model_path)
                print(f"✅ Loaded shape model: {shape_model_path}")
            except Exception as e:
                print(f"❌ Error loading shape model: {e}")
                self.shape_model = None
        else:
            print(f"⚠️ Shape model not found: {shape_model_path}")
            self.shape_model = None
        
        # Set initial model - need to set combobox value first
        self.mode_var.set('🔢 Digit (MNIST)')
        self.on_mode_change()
    
    def on_mode_change(self, event=None):
        """Callback when user changes mode (mnist/shapes)."""
        selected = self.mode_var.get()
        
        # Determine mode from selection
        if 'Digit' in selected or 'MNIST' in selected:
            new_mode = 'mnist'
            self.mode_var.set('🔢 Digit (MNIST)')
        else:
            new_mode = 'shapes'
            self.mode_var.set('🔷 Shape')
        
        self.mode = new_mode

        # Update class names depending on mode
        if new_mode == 'mnist':
            self.class_names = {i: str(i) for i in range(10)}
            self.model = self.digit_model
            if self.model:
                self.model_status_label.config(text="✅ Digit model loaded", fg='#27ae60')
            else:
                self.model_status_label.config(text="❌ Digit model not found", fg='#e74c3c')
        elif new_mode == 'shapes':
            self.class_names = {i: self.shape_names[i] for i in range(len(self.shape_names))}
            self.model = self.shape_model
            if self.model:
                self.model_status_label.config(text="✅ Shape model loaded", fg='#9b59b6')
            else:
                self.model_status_label.config(text="❌ Shape model not found", fg='#e74c3c')


    def preprocess_image(self, img_array):
        """
        Preprocess image using the same pipeline as training
        Returns: 28x28x1 normalized array
        """
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = rgb_to_grayscale(img_array)
            else:
                gray = img_array

            # Threshold & Invert
            thresh = apply_threshold_inverted(gray)

            # Find bounding box and crop
            x_min, y_min, x_max, y_max = find_bounding_box(thresh)
            cropped = crop_image(thresh, x_min, y_min, x_max, y_max)

            # Padding to square
            padded_square = add_padding_square(cropped)

            # Resize to 20x20
            resized_20 = resize_image(padded_square, 20)

            # Add border padding to 28x28
            final_28 = add_border_padding(resized_20, padding=4)

            # Normalize
            normalized = normalize_image(final_28)

            return normalized.reshape(28, 28, 1)

        except ValueError as e:
            # If preprocessing fails, use simple resize
            print(f"Preprocessing failed: {e}. Using simple resize.")
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(
                img_array.shape) == 3 else img_array
            resized = cv2.resize(gray, (28, 28))
            normalized = resized.astype('float32') / 255.0
            return normalized.reshape(28, 28, 1)


    def upload_image(self):
        """Upload and predict image from file"""
        if self.model is None:
            messagebox.showwarning("Warning", "No model loaded!")
            return

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            # Load image
            img = Image.open(file_path)

            # Show preview
            img_preview = img.copy()
            img_preview.thumbnail((150, 150))
            photo = ImageTk.PhotoImage(img_preview)
            self.upload_image_label.config(image=photo)
            self.upload_image_label.image = photo
            self.upload_preview_label.config(
                text=f"Loaded: {os.path.basename(file_path)}"
            )

            # Convert to array
            img_array = np.array(img)
            self.last_uploaded_image = img_array.copy()  # Store for visualization

            # Preprocess
            processed = self.preprocess_image(img_array)

            # Show processed image
            self.show_processed_image(processed)

            # Predict
            self.make_prediction(processed)
            
            # Enable show steps button
            self.show_steps_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def show_processed_image(self, processed_array):
        """Display processed 28x28 image"""
        # Convert to 0-255 range
        img_display = (processed_array.reshape(28, 28) * 255).astype(np.uint8)

        # Resize for display
        img_display = cv2.resize(
            img_display, (140, 140), interpolation=cv2.INTER_NEAREST)

        # Convert to PIL
        img_pil = Image.fromarray(img_display)
        photo = ImageTk.PhotoImage(img_pil)

        self.processed_image_label.config(image=photo)
        self.processed_image_label.image = photo

    def make_prediction(self, processed_array):
        """Make prediction and display results"""
        # Expand dims for batch
        input_array = np.expand_dims(processed_array, axis=0)

        # Predict
        predictions = self.model.predict(input_array, verbose=0)[0]

        # Get top prediction
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        class_name = self.class_names[predicted_class]
        display_name = self.get_display_name(class_name)

        # Update result label
        self.result_label.config(
            text=f"Dự đoán: {display_name}\nĐộ tin cậy: {confidence*100:.2f}%",
            fg='#27ae60' if confidence > 0.7 else '#e67e22',
            font=('Arial', 16, 'bold')
        )

        # Show top 3 predictions
        self.show_top_predictions(predictions)

    def show_top_predictions(self, predictions):
        """Display top 3 predictions"""
        # Clear previous
        for widget in self.top3_frame.winfo_children():
            widget.destroy()

        # Get top 3
        top3_indices = np.argsort(predictions)[-3:][::-1]

        tk.Label(
            self.top3_frame,
            text="Top 3 dự đoán:",
            font=('Arial', 11, 'bold'),
            bg='#ecf0f1'
        ).pack()

        for i, idx in enumerate(top3_indices, 1):
            class_name = self.class_names[idx]
            display_name = self.get_display_name(class_name)
            confidence = predictions[idx]

            # Color based on rank
            colors = ['#27ae60', '#3498db', '#95a5a6']

            pred_frame = tk.Frame(self.top3_frame, bg='#ecf0f1')
            pred_frame.pack(pady=2)

            tk.Label(
                pred_frame,
                text=f"{i}. {display_name}: {confidence*100:.2f}%",
                font=('Arial', 10),
                bg='#ecf0f1',
                fg=colors[i-1]
            ).pack()

    def setup_air_writing_tab(self):
        """Setup tab Air Writing"""
        # Control panel
        control_frame = tk.Frame(self.air_writing_frame, bg='#ecf0f1')
        control_frame.pack(fill=tk.X, pady=15)
        
        start_btn = tk.Button(
            control_frame,
            text="🎥 Bắt đầu Air Writing",
            command=self.start_air_writing,
            bg='#27ae60',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=25,
            pady=12,
            cursor='hand2'
        )
        start_btn.pack(side=tk.LEFT, padx=8)
        
        stop_btn = tk.Button(
            control_frame,
            text="⏹️ Dừng",
            command=self.stop_air_writing,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=25,
            pady=12,
            cursor='hand2'
        )
        stop_btn.pack(side=tk.LEFT, padx=8)
        
        # Instructions
        instr_label = tk.Label(
            control_frame,
            text="Hướng dẫn: Nhấn 'd' để bật/tắt vẽ | 'c' để xóa | 'p' để nhận diện | 'q' để thoát",
            bg='#ecf0f1',
            fg='#e67e22',
            font=('Arial', 10, 'bold')
        )
        instr_label.pack(side=tk.LEFT, padx=25)
        
        # Main content - split view (Camera lớn, Results nhỏ)
        content_frame = tk.Frame(self.air_writing_frame, bg='#ecf0f1')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left - Camera (chiếm 70% không gian)
        left_frame = tk.LabelFrame(
            content_frame,
            text="📹 Camera",
            font=('Arial', 14, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera label với kích thước lớn hơn
        self.camera_label = tk.Label(
            left_frame,
            bg='#1a1a1a',
            text="Nhấn 'Bắt đầu Air Writing' để mở camera",
            font=('Arial', 14),
            fg='#ffffff',
            anchor=tk.CENTER
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bind keyboard events for Air Writing
        self.root.bind('<KeyPress>', self.handle_air_writing_key)
        self.camera_label.bind('<Button-1>', lambda e: self.root.focus_set())
        
        # Right - Results (chiếm 30% không gian, cố định chiều rộng)
        right_frame = tk.Frame(
            content_frame,
            bg='#ffffff',
            relief=tk.RAISED,
            bd=2
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.config(width=350)  # Cố định chiều rộng
        
        # Results header
        results_header = tk.Frame(right_frame, bg='#3498db', height=50)
        results_header.pack(fill=tk.X)
        results_header.pack_propagate(False)
        
        tk.Label(
            results_header,
            text="🎯 Kết quả nhận diện",
            font=('Arial', 13, 'bold'),
            bg='#3498db',
            fg='white'
        ).pack(pady=15)
        
        # Show preprocessing steps button - ĐẶT Ở TRÊN, SAU HEADER
        self.air_show_steps_btn = tk.Button(
            right_frame,
            text="🔬 Hiển thị quá trình xử lý",
            command=self.show_preprocessing_steps_air_writing,
            font=('Arial', 12, 'bold'),
            bg='#9b59b6',
            fg='white',
            padx=20,
            pady=12,
            cursor='hand2',
            state=tk.DISABLED,
            relief=tk.FLAT,
            bd=0,
            activebackground='#8e44ad',
            activeforeground='white'
        )
        self.air_show_steps_btn.pack(pady=(10, 10), padx=10, fill=tk.X)
        
        # Results content area - ĐẶT SAU NÚT
        results_content = tk.Frame(right_frame, bg='#ffffff')
        results_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Scrollable text widget for results
        result_scroll = tk.Scrollbar(results_content, bg='#f0f0f0', troughcolor='#ffffff')
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.air_result_label = tk.Text(
            results_content,
            bg='#fafafa',
            fg='#2c3e50',
            font=('Segoe UI', 10),
            wrap=tk.WORD,
            yscrollcommand=result_scroll.set,
            padx=12,
            pady=12,
            relief=tk.FLAT,
            bd=0,
            selectbackground='#3498db'
        )
        self.air_result_label.pack(fill=tk.BOTH, expand=True)
        self.air_result_label.insert('1.0', "Chưa có kết quả")
        self.air_result_label.config(state=tk.DISABLED)
        result_scroll.config(command=self.air_result_label.yview)

    def on_tab_change(self, event=None):
        """Xử lý khi chuyển tab"""
        selected_tab = self.notebook.index(self.notebook.select())
        if selected_tab == 1:  # Air Writing tab
            if self.air_writing_core.active:
                self.root.focus_set()
                self.root.focus_force()

    def start_air_writing(self):
        """Bắt đầu Air Writing"""
        if not self.air_writing_core.start():
            messagebox.showerror("Lỗi", "Không thể mở camera!")
            return
        
        self.root.focus_set()
        self.root.focus_force()
        
        # Start camera thread
        threading.Thread(target=self.air_writing_loop, daemon=True).start()

    def stop_air_writing(self):
        """Dừng Air Writing"""
        self.air_writing_core.stop()
        self.camera_label.configure(
            image="",
            text="Nhấn 'Bắt đầu Air Writing' để mở camera",
            bg='#1a1a1a',
            fg='#ffffff'
        )
        self.camera_label.image = None

    def air_writing_loop(self):
        """Vòng lặp chính cho Air Writing - tối ưu cho mượt mà và phản hồi nhanh"""
        target_fps = 60  # Tăng lên 60 FPS để phản hồi nhanh hơn
        frame_time = 1.0 / target_fps
        
        while self.air_writing_core.active:
            start_time = time.time()
            
            display = self.air_writing_core.process_frame()
            if display is None:
                break
            
            # Display in GUI - tối ưu resize
            display_rgb = bgr_to_rgb(display)
            display_pil = Image.fromarray(display_rgb)
            # Chỉ resize nếu cần (giảm overhead)
            display_pil = display_pil.resize((640, 480), Image.LANCZOS)
            display_tk = ImageTk.PhotoImage(image=display_pil)
            
            self.camera_label.configure(image=display_tk, text="")
            self.camera_label.image = display_tk
            
            # Frame rate control - đảm bảo 30 FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def predict_air_writing(self):
        """Dự đoán từ canvas Air Writing"""
        canvas = self.air_writing_core.get_canvas()
        if canvas is None or np.sum(canvas) == 0:
            messagebox.showwarning("Cảnh báo", "Vui lòng vẽ trước!")
            return
        
        if self.model is None:
            messagebox.showerror("Lỗi", "Model chưa được load!")
            return
        
        try:
            # Store canvas for visualization
            self.last_air_writing_canvas = canvas.copy()
            
            # Preprocess canvas (grayscale, nền đen, nét trắng)
            # Canvas đã là grayscale, cần convert sang RGB format để dùng rgb_to_grayscale
            # Nhưng canvas là grayscale rồi, nên có thể dùng trực tiếp
            processed = self.preprocess_image(canvas)
            
            # Predict
            input_array = np.expand_dims(processed, axis=0)
            predictions = self.model.predict(input_array, verbose=0)[0]
            
            # Get top prediction
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            class_name = self.class_names[predicted_class]
            display_name = self.get_display_name(class_name)
            
            # Format result
            result_text = f"Dự đoán: {display_name}\n"
            result_text += f"Độ tin cậy: {confidence*100:.2f}%\n\n"
            result_text += "Top 3 dự đoán:\n"
            
            # Top 3
            top3_indices = np.argsort(predictions)[-3:][::-1]
            for i, idx in enumerate(top3_indices, 1):
                class_name_item = self.class_names[idx]
                display_name_item = self.get_display_name(class_name_item)
                result_text += f"{i}. {display_name_item}: {predictions[idx]*100:.2f}%\n"
            
            self.air_result_label.config(state=tk.NORMAL)
            self.air_result_label.delete('1.0', tk.END)
            self.air_result_label.insert('1.0', result_text)
            self.air_result_label.config(
                state=tk.DISABLED,
                fg='#27ae60' if confidence > 0.7 else '#e67e22'
            )
            
            # Enable show steps button
            self.air_show_steps_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán: {e}")

    def handle_air_writing_key(self, event):
        """Xử lý phím bấm cho Air Writing"""
        selected_tab = self.notebook.index(self.notebook.select())
        if selected_tab != 1 or not self.air_writing_core.active:
            return
        
        key = event.char.lower() if event.char else ''
        keysym = event.keysym.lower()
        
        if key == 'q' or keysym == 'q':
            self.stop_air_writing()
        elif key == 'd' or keysym == 'd':
            self.air_writing_core.drawing = not self.air_writing_core.drawing
            if not self.air_writing_core.drawing:
                self.air_writing_core.prev_point = None
        elif key == 'c' or keysym == 'c':
            self.air_writing_core.clear_canvas()
            self.air_result_label.config(state=tk.NORMAL)
            self.air_result_label.delete('1.0', tk.END)
            self.air_result_label.insert('1.0', "Chưa có kết quả")
            self.air_result_label.config(fg='#2c3e50', state=tk.DISABLED)
        elif key == 'p' or keysym == 'p':
            self.predict_air_writing()

    def get_preprocessing_steps(self, image_array):
        """Lấy các bước preprocessing để hiển thị"""
        try:
            steps = {}
            
            # Original (convert to grayscale for display if needed)
            if len(image_array.shape) == 3:
                original_display = rgb_to_grayscale(image_array)
            else:
                original_display = image_array.copy()
            if original_display.dtype != np.uint8:
                if original_display.max() <= 1.0:
                    original_display = (original_display * 255).astype(np.uint8)
                else:
                    original_display = original_display.astype(np.uint8)
            steps['1. Original'] = original_display
            
            # Grayscale
            if len(image_array.shape) == 3:
                gray = rgb_to_grayscale(image_array)
            else:
                gray = image_array.copy()
            if gray.dtype != np.uint8:
                if gray.max() <= 1.0:
                    gray = (gray * 255).astype(np.uint8)
                else:
                    gray = gray.astype(np.uint8)
            steps['2. Grayscale'] = gray
            
            # Threshold & Invert
            thresh = apply_threshold_inverted(gray)
            steps['3. Threshold & Invert'] = thresh
            
            # Crop
            try:
                x_min, y_min, x_max, y_max = find_bounding_box(thresh)
                cropped = crop_image(thresh, x_min, y_min, x_max, y_max)
                steps['4. Cropped'] = cropped
            except ValueError:
                steps['4. Cropped'] = thresh  # Fallback
            
            # Square Padding
            cropped_img = steps.get('4. Cropped', thresh)
            padded_square = add_padding_square(cropped_img)
            steps['5. Square Padding'] = padded_square
            
            # Resize to 20x20
            resized_20 = resize_image(padded_square, 20)
            steps['6. Resized 20x20'] = resized_20
            
            # Border Padding to 28x28
            final_28 = add_border_padding(resized_20, padding=4)
            steps['7. Final 28x28'] = final_28
            
            return steps
        except Exception as e:
            print(f"Error getting preprocessing steps: {e}")
            return None

    def show_preprocessing_steps_upload(self):
        """Hiển thị quá trình xử lý cho ảnh upload"""
        if self.last_uploaded_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh để hiển thị!")
            return
        
        steps = self.get_preprocessing_steps(self.last_uploaded_image)
        if steps is None:
            messagebox.showerror("Lỗi", "Không thể lấy các bước xử lý!")
            return
        
        self.show_steps_window(steps)

    def show_preprocessing_steps_air_writing(self):
        """Hiển thị quá trình xử lý cho Air Writing canvas"""
        if self.last_air_writing_canvas is None:
            messagebox.showwarning("Cảnh báo", "Chưa có canvas để hiển thị!")
            return
        
        steps = self.get_preprocessing_steps(self.last_air_writing_canvas)
        if steps is None:
            messagebox.showerror("Lỗi", "Không thể lấy các bước xử lý!")
            return
        
        self.show_steps_window(steps)

    def show_steps_window(self, steps):
        """Hiển thị cửa sổ các bước preprocessing"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            messagebox.showerror("Lỗi", "Cần cài đặt matplotlib để hiển thị!")
            return
        
        # Create new window
        window = tk.Toplevel(self.root)
        window.title("Quá trình xử lý ảnh")
        window.geometry("1200x700")
        window.configure(bg='#ffffff')
        
        # Create figure
        fig = plt.figure(figsize=(14, 7), facecolor='#ffffff')
        canvas_fig = FigureCanvasTkAgg(fig, window)
        canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Plot steps
        n_steps = len(steps)
        cols = min(4, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        for idx, (name, img) in enumerate(steps.items()):
            ax = fig.add_subplot(rows, cols, idx + 1, facecolor='#ffffff')
            ax.set_title(name, color='#2c3e50', fontsize=11, fontweight='bold', pad=8)
            ax.axis('off')
            
            if len(img.shape) == 3:
                ax.imshow(bgr_to_rgb(img))
            else:
                ax.imshow(img, cmap='gray')
        
        fig.tight_layout(pad=3.0)
        canvas_fig.draw()


def main():
    """Main function"""
    root = tk.Tk()
    # No default combined model; let user pick
    app = PredictionGUI(root, None)
    root.mainloop()


if __name__ == "__main__":
    main()
