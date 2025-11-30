"""
Module xử lý ảnh cho MNIST Custom Dataset
Chứa các hàm preprocessing để chuyển đổi ảnh về định dạng 28x28 grayscale
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def rgb_to_grayscale(image_array):
    """
    Chuyển ảnh RGB sang grayscale bằng công thức chuẩn
    Gray = 0.299*R + 0.587*G + 0.114*B

    Args:
        image_array: Ảnh RGB hoặc grayscale dạng numpy array

    Returns:
        Ảnh grayscale dạng numpy array (uint8)
    """
    # Handle already-grayscale
    if image_array is None:
        raise ValueError("Input image is None")

    if len(image_array.shape) == 2:
        gray = image_array
    else:
        # If image has alpha channel, drop it
        if image_array.shape[-1] == 4:
            image_array = image_array[..., :3]

        # If values are floats in [0,1], convert to 0-255
        if np.issubdtype(image_array.dtype, np.floating):
            if image_array.max() <= 1.0:
                image_array = (image_array * 255.0).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)

        gray = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])

    # Ensure uint8 output
    gray = np.clip(gray, 0, 255)
    return gray.astype(np.uint8)


def otsu_threshold(gray_image):
    """
    Tính ngưỡng tối ưu bằng phương pháp Otsu

    Args:
        gray_image: Ảnh grayscale dạng numpy array

    Returns:
        Giá trị ngưỡng tối ưu (int)
    """
    hist, _ = np.histogram(gray_image.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    hist /= hist.sum()

    bins = np.arange(256)
    weight1 = np.cumsum(hist)
    weight2 = 1.0 - weight1

    mean1 = np.cumsum(hist * bins) / (weight1 + 1e-10)
    global_mean = np.sum(hist * bins)
    mean2 = (global_mean - np.cumsum(hist * bins)) / (weight2 + 1e-10)

    variance = weight1 * weight2 * (mean1 - mean2) ** 2
    threshold = np.argmax(variance)

    return threshold


def apply_threshold_inverted(gray_image):
    """
    Áp dụng threshold và đảo ngược (nền đen, chữ trắng)

    Args:
        gray_image: Ảnh grayscale dạng numpy array

    Returns:
        Ảnh binary inverted (uint8)
    """
    # Compute Otsu threshold
    threshold = otsu_threshold(gray_image)

    # Candidate 1: non-inverted (foreground = bright regions)
    binary_noninv = np.zeros_like(gray_image, dtype=np.uint8)
    binary_noninv[gray_image >= threshold] = 255
    binary_noninv[gray_image < threshold] = 0

    # Candidate 2: inverted (foreground = dark regions)
    binary_inv = np.zeros_like(gray_image, dtype=np.uint8)
    binary_inv[gray_image < threshold] = 255
    binary_inv[gray_image >= threshold] = 0

    # Evaluate candidates by fraction of foreground pixels
    total_pixels = gray_image.size
    cnt_noninv = np.count_nonzero(binary_noninv)
    cnt_inv = np.count_nonzero(binary_inv)
    frac_noninv = cnt_noninv / total_pixels
    frac_inv = cnt_inv / total_pixels

    # Prefer a candidate whose foreground is small but non-zero (object occupies small area)
    min_frac = 1e-4
    max_frac = 0.5

    valid_noninv = (cnt_noninv > 0) and (min_frac <= frac_noninv <= max_frac)
    valid_inv = (cnt_inv > 0) and (min_frac <= frac_inv <= max_frac)

    if valid_noninv and valid_inv:
        # Both look plausible: pick the smaller foreground (likely the object)
        chosen = binary_noninv if frac_noninv <= frac_inv else binary_inv
        return chosen

    if valid_noninv:
        return binary_noninv

    if valid_inv:
        return binary_inv

    # If neither candidate is in the desired fraction range, try mean-based threshold
    alt_thresh = int(np.mean(gray_image))
    alt_bin = np.zeros_like(gray_image, dtype=np.uint8)
    alt_bin[gray_image >= alt_thresh] = 255
    alt_bin[gray_image < alt_thresh] = 0
    if np.count_nonzero(alt_bin) > 0 and (np.count_nonzero(alt_bin) / total_pixels) <= max_frac:
        return alt_bin

    # Last resort: treat non-white pixels as foreground but try to avoid full-image masks
    mask = gray_image < 250
    fallback = np.zeros_like(gray_image, dtype=np.uint8)
    fallback[mask] = 255
    if np.count_nonzero(fallback) == 0 or (np.count_nonzero(fallback) / total_pixels) > 0.95:
        # If it's still bad, return the non-inverted candidate (safer default)
        return binary_noninv

    return fallback


def find_bounding_box(binary_image):
    """
    Tìm bounding box của vùng có pixel trắng (chữ số)

    Args:
        binary_image: Ảnh binary dạng numpy array

    Returns:
        Tuple (x_min, y_min, x_max, y_max)

    Raises:
        ValueError: Nếu không tìm thấy vùng chữ số
    """
    rows = np.any(binary_image, axis=1)
    cols = np.any(binary_image, axis=0)

    if not np.any(rows) or not np.any(cols):
        raise ValueError("Không tìm thấy vùng chữ số trong ảnh")

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return x_min, y_min, x_max, y_max


def crop_image(image, x_min, y_min, x_max, y_max):
    """
    Crop ảnh theo bounding box

    Args:
        image: Ảnh dạng numpy array
        x_min, y_min, x_max, y_max: Tọa độ bounding box

    Returns:
        Ảnh đã crop
    """
    return image[y_min:y_max+1, x_min:x_max+1]


def add_padding_square(image):
    """
    Thêm padding để ảnh thành hình vuông

    Args:
        image: Ảnh dạng numpy array

    Returns:
        Ảnh vuông với padding
    """
    h, w = image.shape

    if h == w:
        return image

    size = max(h, w)
    padded = np.zeros((size, size), dtype=image.dtype)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    padded[y_offset:y_offset+h, x_offset:x_offset+w] = image

    return padded


def resize_image(image, new_size):
    """
    Resize ảnh bằng PIL

    Args:
        image: Ảnh dạng numpy array
        new_size: Kích thước mới (int)

    Returns:
        Ảnh đã resize (numpy array)
    """
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((new_size, new_size), Image.LANCZOS)
    return np.array(resized)


def add_border_padding(image, padding=4):
    """
    Thêm padding xung quanh ảnh

    Args:
        image: Ảnh dạng numpy array
        padding: Số pixel padding (mặc định 4)

    Returns:
        Ảnh với border padding
    """
    h, w = image.shape
    padded = np.zeros((h + 2*padding, w + 2*padding), dtype=image.dtype)
    padded[padding:padding+h, padding:padding+w] = image
    return padded


def normalize_image(image):
    """
    Normalize ảnh về khoảng [0.0, 1.0]

    Args:
        image: Ảnh dạng numpy array (uint8)

    Returns:
        Ảnh normalized (float32)
    """
    return image.astype(np.float32) / 255.0


def preprocess_image_simple(img_path):
    """
    Pipeline đầy đủ: Load → Grayscale → Threshold → Crop → Pad → Resize → Normalize

    Args:
        img_path: Đường dẫn đến file ảnh

    Returns:
        Ảnh đã xử lý (28x28, normalized [0.0, 1.0])
    """
    # Load ảnh
    pil_image = Image.open(img_path)
    original = np.array(pil_image)

    # 1. Grayscale
    gray = rgb_to_grayscale(original)

    # 2. Threshold & Invert
    thresh = apply_threshold_inverted(gray)

    # 3. Crop
    x_min, y_min, x_max, y_max = find_bounding_box(thresh)
    cropped = crop_image(thresh, x_min, y_min, x_max, y_max)

    # 4. Padding & Resize
    padded_square = add_padding_square(cropped)
    resized_20 = resize_image(padded_square, 20)
    final_28 = add_border_padding(resized_20, padding=4)

    # 5. Normalize
    normalized = normalize_image(final_28)

    return normalized


def preprocess_image_array(image_array):
    """
    Xử lý ảnh từ numpy array (thay vì đường dẫn file)

    Args:
        image_array: Ảnh dạng numpy array (RGB hoặc grayscale)

    Returns:
        Ảnh đã xử lý (28x28, normalized [0.0, 1.0])
    """
    # 1. Grayscale
    gray = rgb_to_grayscale(image_array)

    # 2. Threshold & Invert
    thresh = apply_threshold_inverted(gray)

    # 3. Crop
    x_min, y_min, x_max, y_max = find_bounding_box(thresh)
    cropped = crop_image(thresh, x_min, y_min, x_max, y_max)

    # 4. Padding & Resize
    padded_square = add_padding_square(cropped)
    resized_20 = resize_image(padded_square, 20)
    final_28 = add_border_padding(resized_20, padding=4)

    # 5. Normalize
    normalized = normalize_image(final_28)

    return normalized


def visualize_preprocessing_steps(img_path):
    """
    Hiển thị từng bước preprocessing trên một ảnh (cần matplotlib)

    Args:
        img_path: Đường dẫn đến file ảnh
    """
    import matplotlib.pyplot as plt

    # Load ảnh gốc
    pil_image = Image.open(img_path)
    original = np.array(pil_image)

    # Bước 1: Grayscale
    gray = rgb_to_grayscale(original)

    # Bước 2: Threshold & Invert
    thresh = apply_threshold_inverted(gray)

    # Bước 3: Crop
    x_min, y_min, x_max, y_max = find_bounding_box(thresh)
    cropped = crop_image(thresh, x_min, y_min, x_max, y_max)

    # Bước 4: Padding & Resize
    padded_square = add_padding_square(cropped)
    resized_20 = resize_image(padded_square, 20)

    # Bước 5: Border padding + Normalize
    final_28 = add_border_padding(resized_20, padding=4)
    normalized = normalize_image(final_28)

    # Hiển thị
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1
    axes[0, 0].imshow(original)
    axes[0, 0].set_title(
        f'1. Original\nSize: {original.shape}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title(
        f'2. Grayscale\nSize: {gray.shape}', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(thresh, cmap='gray')
    axes[0, 2].set_title(
        f'3. Threshold + Invert\nSize: {thresh.shape}', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].imshow(cropped, cmap='gray')
    axes[1, 0].set_title(
        f'4. Cropped\nSize: {cropped.shape}', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(padded_square, cmap='gray')
    axes[1, 1].set_title(f'5. Padding + Resize 20x20\nSize: {padded_square.shape} → (20, 20)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(normalized, cmap='gray')
    axes[1, 2].set_title(f'6. Final (Border + Normalize)\nSize: {normalized.shape}\nRange: [0.0, 1.0]',
                         fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # In thông tin
    print(f"{'='*70}")
    print(f"Thông tin chi tiết:")
    print(f"  Original size: {original.shape}")
    print(f"  After grayscale: {gray.shape}")
    print(f"  After threshold: {thresh.shape}")
    print(f"  After crop: {cropped.shape}")
    print(f"  After padding square: {padded_square.shape}")
    print(f"  After resize to 20x20: (20, 20)")
    print(f"  After border padding: {final_28.shape}")
    print(
        f"  After normalize: {normalized.shape}, range [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"{'='*70}")


# ========== Helper functions for Air Writing ==========

def bgr_to_rgb(image):
    """
    Chuyển đổi ảnh từ BGR sang RGB
    Args:
        image: Ảnh BGR (H, W, 3)
    Returns:
        Ảnh RGB (H, W, 3)
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
    # BGR -> RGB: đảo ngược channel cuối cùng
    return image[:, :, ::-1].copy()


def grayscale_to_bgr(image):
    """
    Chuyển đổi ảnh grayscale sang BGR (3 channels)
    Args:
        image: Ảnh grayscale (H, W)
    Returns:
        Ảnh BGR (H, W, 3)
    """
    if len(image.shape) == 2:
        # Grayscale -> BGR: copy channel 3 lần
        return np.stack([image, image, image], axis=2)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # (H, W, 1) -> (H, W, 3)
        return np.repeat(image, 3, axis=2)
    return image.copy()


def flip_horizontal(image):
    """
    Lật ảnh theo chiều ngang (trái -> phải)
    Args:
        image: Ảnh đầu vào
    Returns:
        Ảnh đã lật
    """
    return np.fliplr(image).copy()


def draw_circle(image, center, radius, color, thickness=-1):
    img = image.copy()
    x, y = center
    
    h, w = img.shape[:2]
    
    # Tạo mask cho hình tròn
    yy, xx = np.ogrid[:h, :w]
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    
    if thickness == -1:
        # Fill circle
        mask = dist_sq <= radius ** 2
    else:
        # Circle outline
        outer_mask = dist_sq <= radius ** 2
        inner_mask = dist_sq <= (radius - thickness) ** 2
        mask = outer_mask & ~inner_mask
    
    # Vẽ màu
    if len(img.shape) == 2:
        # Grayscale
        if isinstance(color, (int, np.integer)):
            img[mask] = color
        else:
            # Nếu color là tuple, lấy giá trị đầu tiên
            img[mask] = color[0] if len(color) > 0 else 255
    else:
        # Color image
        if isinstance(color, (int, np.integer)):
            img[mask] = color
        else:
            # BGR format
            for c in range(min(len(color), img.shape[2])):
                img[:, :, c][mask] = color[c]
    
    return img


def draw_line(image, pt1, pt2, color, thickness=1):
    img = image.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    
    h, w = img.shape[:2]
    
    # Tính toán các điểm trên đường thẳng bằng Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    
    # Tập hợp tất cả các điểm trên đường thẳng
    points = []
    while True:
        points.append((x, y))
        if x == x2 and y == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    # Vẽ các điểm với thickness
    if thickness == 1:
        # Vẽ từng điểm
        for px, py in points:
            if 0 <= px < w and 0 <= py < h:
                if len(img.shape) == 2:
                    # Grayscale
                    if isinstance(color, (int, np.integer)):
                        img[py, px] = color
                    else:
                        img[py, px] = color[0] if len(color) > 0 else 255
                else:
                    # Color
                    if isinstance(color, (int, np.integer)):
                        img[py, px] = color
                    else:
                        for c in range(min(len(color), img.shape[2])):
                            img[py, px, c] = color[c]
    else:
        # Vẽ với thickness > 1: vẽ nhiều điểm xung quanh mỗi điểm trên đường thẳng
        radius = thickness // 2
        for px, py in points:
            # Vẽ hình tròn nhỏ tại mỗi điểm
            yy, xx = np.ogrid[:h, :w]
            dist_sq = (xx - px) ** 2 + (yy - py) ** 2
            mask = dist_sq <= radius ** 2
            
            if len(img.shape) == 2:
                if isinstance(color, (int, np.integer)):
                    img[mask] = color
                else:
                    img[mask] = color[0] if len(color) > 0 else 255
            else:
                if isinstance(color, (int, np.integer)):
                    img[mask] = color
                else:
                    for c in range(min(len(color), img.shape[2])):
                        img[:, :, c][mask] = color[c]
    
    return img


def add_weighted(src1, alpha, src2, beta, gamma=0.0):
    # Đảm bảo 2 ảnh cùng kích thước
    if src1.shape != src2.shape:
        # Resize src2 về kích thước src1
        h, w = src1.shape[:2]
        src2_resized = resize_image(src2, max(h, w))
        # Crop nếu cần
        if src2_resized.shape[:2] != (h, w):
            src2_resized = src2_resized[:h, :w]
        src2 = src2_resized
    
    # Trộn ảnh
    result = alpha * src1.astype(np.float32) + beta * src2.astype(np.float32) + gamma
    
    # Clip về [0, 255] và chuyển về uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def put_text(image, text, org, font_scale=1.0, color=(255, 255, 255), thickness=1, font=None):
    img = image.copy()
    x, y = org
    
    # Chuyển sang PIL để vẽ text dễ hơn
    if len(img.shape) == 2:
        # Grayscale -> RGB
        pil_image = Image.fromarray(img, mode='L').convert('RGB')
        is_grayscale = True
    else:
        # BGR -> RGB
        pil_image = Image.fromarray(bgr_to_rgb(img), mode='RGB')
        is_grayscale = False
    
    # Tạo ImageDraw
    draw = ImageDraw.Draw(pil_image)
    
    # Tính kích thước font
    try:
        # Thử dùng font hệ thống
        if font and os.path.exists(font):
            base_font = ImageFont.truetype(font, int(20 * font_scale))
        else:
            # Font mặc định
            try:
                # Thử dùng font mặc định của hệ thống
                if os.name == 'nt':  # Windows
                    base_font = ImageFont.truetype("arial.ttf", int(20 * font_scale))
                else:  # Linux/Mac
                    base_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(20 * font_scale))
            except:
                # Fallback về font mặc định
                base_font = ImageFont.load_default()
    except:
        base_font = ImageFont.load_default()
    
    # Chuyển màu BGR -> RGB
    if isinstance(color, (int, np.integer)):
        text_color = (color, color, color)
    else:
        if len(color) >= 3:
            # BGR -> RGB
            text_color = (color[2], color[1], color[0])
        else:
            text_color = (255, 255, 255)
    
    # Vẽ text
    # PIL dùng (x, y) là góc trên bên trái, cần điều chỉnh
    try:
        bbox = draw.textbbox((x, y), text, font=base_font)
        text_height = bbox[3] - bbox[1]
        # Điều chỉnh y để text bắt đầu từ org (góc dưới bên trái)
        text_y = max(0, y - text_height)  # Đảm bảo không âm
    except:
        # Fallback nếu không đo được
        text_y = max(0, y - 20)
    
    # Vẽ text với outline nếu thickness > 1
    if thickness > 1:
        # Vẽ outline
        for adj in range(-thickness, thickness + 1):
            for adj2 in range(-thickness, thickness + 1):
                if adj != 0 or adj2 != 0:
                    draw.text((x + adj, text_y + adj2), text, font=base_font, fill=(0, 0, 0))
    
    # Vẽ text chính
    draw.text((x, text_y), text, font=base_font, fill=text_color)
    
    # Chuyển về numpy array
    result_array = np.array(pil_image)
    
    # Chuyển về BGR nếu cần
    if len(img.shape) == 3:
        result_array = result_array[:, :, ::-1]  # RGB -> BGR
    
    # Chuyển về grayscale nếu ảnh gốc là grayscale
    if is_grayscale:
        result_array = rgb_to_grayscale(result_array)
    
    return result_array


# Hàm main để test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_processing.py <image_path>")
        print("Example: python image_processing.py test_image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]

    try:
        # Xử lý ảnh
        print(f"Processing image: {img_path}")
        processed = preprocess_image_simple(img_path)

        print(f"✓ Image processed successfully!")
        print(f"  Output shape: {processed.shape}")
        print(f"  Value range: [{processed.min():.3f}, {processed.max():.3f}]")

        # Hiển thị chi tiết (nếu có matplotlib)
        try:
            visualize_preprocessing_steps(img_path)
        except ImportError:
            print("\nNote: Install matplotlib to visualize preprocessing steps")
            print("  pip install matplotlib")

    except Exception as e:
        print(f"✗ Error processing image: {e}")
        sys.exit(1)
