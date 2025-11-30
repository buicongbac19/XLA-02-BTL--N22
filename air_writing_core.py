"""
Module xử lý core logic cho Air Writing với smoothing
"""
import cv2  # Chỉ dùng cho VideoCapture
import numpy as np
import mediapipe as mp
import time

# Cấu hình Air Writing
SMOOTHING_FACTOR = 0.75  # Hệ số smoothing cơ bản (0-1), giảm để phản hồi nhanh hơn
MIN_DISTANCE = 3  # Khoảng cách tối thiểu giữa các điểm để vẽ (pixels) - giảm để bắt nhiều điểm hơn
MAX_DISTANCE_FOR_INTERPOLATION = 25  # Khoảng cách tối đa để thêm interpolation (pixels)
FAST_MOVEMENT_THRESHOLD = 15  # Ngưỡng tốc độ di chuyển nhanh (pixels/frame)

from preprocessing.image_processing import (
    flip_horizontal, bgr_to_rgb, grayscale_to_bgr,
    draw_circle, draw_line, add_weighted, put_text
)


class SmoothingFilter:
    """Lớp lọc smoothing cho tọa độ ngón tay với adaptive smoothing"""
    def __init__(self, alpha=0.75):
        """
        alpha: hệ số smoothing cơ bản (0-1)
        - 0: không smoothing (phản hồi tức thì)
        - 1: smoothing tối đa (rất mượt nhưng chậm)
        """
        self.base_alpha = alpha
        self.x = None
        self.y = None
        self.prev_x = None
        self.prev_y = None
        self.velocity = 0.0  # Tốc độ di chuyển
    
    def update(self, x, y):
        """Cập nhật tọa độ với adaptive smoothing dựa trên tốc độ"""
        if self.x is None or self.y is None:
            # Lần đầu tiên, lấy giá trị trực tiếp
            self.x = float(x)
            self.y = float(y)
            self.prev_x = float(x)
            self.prev_y = float(y)
            self.velocity = 0.0
        else:
            # Tính tốc độ di chuyển
            dx = float(x) - self.prev_x
            dy = float(y) - self.prev_y
            current_velocity = np.sqrt(dx**2 + dy**2)
            
            # Smooth velocity để tránh thay đổi đột ngột
            self.velocity = 0.7 * self.velocity + 0.3 * current_velocity
            
            # Adaptive smoothing: di chuyển nhanh thì smoothing ít hơn (phản hồi nhanh)
            # di chuyển chậm thì smoothing nhiều hơn (mượt mà)
            if self.velocity > FAST_MOVEMENT_THRESHOLD:
                # Di chuyển nhanh: giảm smoothing để phản hồi nhanh
                alpha = self.base_alpha * 0.6  # Smoothing ít hơn
            elif self.velocity > FAST_MOVEMENT_THRESHOLD * 0.5:
                # Di chuyển trung bình: smoothing vừa phải
                alpha = self.base_alpha * 0.8
            else:
                # Di chuyển chậm: smoothing nhiều để mượt
                alpha = self.base_alpha
            
            # Exponential smoothing với alpha động
            self.x = alpha * self.x + (1 - alpha) * float(x)
            self.y = alpha * self.y + (1 - alpha) * float(y)
            
            # Lưu tọa độ gốc để tính velocity cho lần sau
            self.prev_x = float(x)
            self.prev_y = float(y)
        
        return int(self.x), int(self.y)
    
    def reset(self):
        """Reset filter"""
        self.x = None
        self.y = None
        self.prev_x = None
        self.prev_y = None
        self.velocity = 0.0


class AirWritingCore:
    """Lớp xử lý core logic cho Air Writing"""
    def __init__(self):
        self.cap = None
        self.active = False
        self.canvas = None
        self.drawing = False
        self.prev_point = None
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = None
        
        # Smoothing filter
        self.smoother = SmoothingFilter(alpha=SMOOTHING_FACTOR)
    
    def start(self, camera_index=0):
        """Bắt đầu Air Writing"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            return False
        
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self.canvas = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        self.active = True
        self.drawing = False
        self.prev_point = None
        self.smoother.reset()
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,  # Giảm để tránh mất detection
            min_tracking_confidence=0.5,    # Giảm để tracking ổn định hơn
            static_image_mode=False,
            model_complexity=0  # Giảm complexity để tăng tốc độ xử lý
        )
        
        return True
    
    def stop(self):
        """Dừng Air Writing"""
        self.active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.hands:
            self.hands.close()
            self.hands = None
        self.smoother.reset()
    
    def process_frame(self):
        """Xử lý một frame và trả về frame đã xử lý"""
        if not self.active or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = flip_horizontal(frame)
        rgb = bgr_to_rgb(frame)
        results = self.hands.process(rgb)
        
        index_point = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_raw = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                y_raw = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                
                # Áp dụng smoothing
                x_smooth, y_smooth = self.smoother.update(x_raw, y_raw)
                index_point = (x_smooth, y_smooth)
                
                # Vẽ chấm đỏ tại ngón tay trỏ (dùng tọa độ đã smooth)
                frame = draw_circle(frame, index_point, 6, (0, 0, 255), -1)
        
        # Drawing logic với kiểm tra khoảng cách và interpolation
        if self.drawing and index_point is not None:
            if self.prev_point is not None:
                # Tính khoảng cách giữa 2 điểm
                dx = index_point[0] - self.prev_point[0]
                dy = index_point[1] - self.prev_point[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                # Chỉ vẽ nếu khoảng cách đủ lớn (giảm rung)
                if distance >= MIN_DISTANCE:
                    # Nếu khoảng cách quá lớn, thêm interpolation để vẽ mượt
                    if distance > MAX_DISTANCE_FOR_INTERPOLATION:
                        # Tính số điểm cần interpolate - tối ưu dựa trên khoảng cách
                        # Di chuyển nhanh: ít điểm hơn để vẽ nhanh, di chuyển chậm: nhiều điểm hơn để mượt
                        segment_size = max(6, min(10, int(distance / 4)))  # Tối ưu segment size
                        num_segments = max(2, int(distance / segment_size))
                        prev_interp_point = self.prev_point
                        
                        for i in range(1, num_segments + 1):
                            t = i / num_segments
                            interp_x = int(self.prev_point[0] + dx * t)
                            interp_y = int(self.prev_point[1] + dy * t)
                            interp_point = (interp_x, interp_y)
                            
                            # Vẽ từng đoạn nhỏ liên tiếp
                            self.canvas = draw_line(self.canvas, prev_interp_point, 
                                                   interp_point, 255, thickness=8)
                            prev_interp_point = interp_point
                    else:
                        # Khoảng cách bình thường, vẽ trực tiếp
                        self.canvas = draw_line(self.canvas, self.prev_point, 
                                               index_point, 255, thickness=8)
                    self.prev_point = index_point
            else:
                # Lần đầu tiên, lưu điểm
                self.prev_point = index_point
        elif not self.drawing:
            # Khi không vẽ, reset prev_point
            self.prev_point = None
        # Nếu drawing=True nhưng không detect được tay,
        # giữ nguyên prev_point để tiếp tục khi detect lại
        
        # Overlay canvas
        color_canvas = grayscale_to_bgr(self.canvas)
        display = add_weighted(frame, 0.7, color_canvas, 0.3, 0)
        
        # Hiển thị trạng thái - font lớn hơn để dễ nhìn
        status_color = (0, 255, 0) if self.drawing else (0, 0, 255)
        display = put_text(display, f"Drawing: {self.drawing}", (10, 50),
                          font_scale=1.5, color=status_color, thickness=3)
        
        return display
    
    def clear_canvas(self):
        """Xóa canvas"""
        if self.canvas is not None:
            self.canvas[:] = 0
        self.prev_point = None
        self.smoother.reset()
    
    def get_canvas(self):
        """Lấy canvas hiện tại"""
        return self.canvas.copy() if self.canvas is not None else None

