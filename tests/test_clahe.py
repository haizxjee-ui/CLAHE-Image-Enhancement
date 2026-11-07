
import numpy as np
import cv2

from algorithms.clahe import (
    clip_histogram,
    create_clahe_lut,
    create_tile_luts,
    clahe
)


# =====================================
# 1. KIEM TRA HISTOGRAM CLIPPING
# =====================================

histogram = np.zeros(256, dtype=np.int64)

histogram[10] = 5
histogram[20] = 10
histogram[30] = 70
histogram[40] = 10
histogram[50] = 5

result_histogram = clip_histogram(histogram, 10)

print("===== HISTOGRAM CLIPPING =====")
print("Tong pixel ban dau:", histogram.sum())
print("Tong pixel sau clipping:", result_histogram.sum())


# =====================================
# 2. KIEM TRA LUT
# =====================================

lut = create_clahe_lut(result_histogram)

print("\n===== KIEM TRA LUT =====")
print("So phan tu LUT:", len(lut))
print("Gia tri nho nhat:", lut.min())
print("Gia tri lon nhat:", lut.max())


# =====================================
# 3. DOC ANH THU NGHIEM
# =====================================

image = cv2.imread(
    "datasets/sample_50/images/00026406_000.png",
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError("Khong tim thay anh thu nghiem")


# =====================================
# 4. KIEM TRA LUT CUA CAC TILE
# =====================================

lut_grid = create_tile_luts(
    image,
    grid_size=(8, 8),
    clip_limit=2.0
)

print("\n===== KIEM TRA TILE LUT =====")
print("Kich thuoc LUT grid:", lut_grid.shape)


# =====================================
# 5. KIEM TRA CLAHE HOAN CHINH
# =====================================

custom_result = clahe(
    image,
    grid_size=(8, 8),
    clip_limit=2.0
)

print("\n===== CLAHE HOAN CHINH =====")
print("Kich thuoc anh:", custom_result.shape)
print("Kieu du lieu:", custom_result.dtype)
print("Gia tri nho nhat:", custom_result.min())
print("Gia tri lon nhat:", custom_result.max())


# =====================================
# 6. SO SANH VOI OPENCV
# =====================================

opencv_clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

opencv_result = opencv_clahe.apply(image)

difference = cv2.absdiff(
    custom_result,
    opencv_result
)

print("\n===== SO SANH VOI OPENCV =====")
print("Sai so lon nhat:", difference.max())
print("Sai so trung binh:", difference.mean())


# =====================================
# 7. HIEN THI KET QUA
# =====================================

cv2.imshow("Original", image)
cv2.imshow("Custom CLAHE", custom_result)
cv2.imshow("OpenCV CLAHE", opencv_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
