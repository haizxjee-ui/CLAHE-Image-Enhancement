import cv2
import matplotlib.pyplot as plt

from algorithms.color_processing import (
    clahe_lab,
    clahe_hsv,
    clahe_rgb
)


# Doc anh mau
image_path = "datasets/sample_color/test.jpg"

image = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image is None:
    raise FileNotFoundError("Khong tim thay anh")


# Ap dung ba phuong phap
result_lab = clahe_lab(image)
result_hsv = clahe_hsv(image)
result_rgb = clahe_rgb(image)


# Chuyen BGR sang RGB de hien thi bang Matplotlib
image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
lab_display = cv2.cvtColor(result_lab, cv2.COLOR_BGR2RGB)
hsv_display = cv2.cvtColor(result_hsv, cv2.COLOR_BGR2RGB)
rgb_display = cv2.cvtColor(result_rgb, cv2.COLOR_BGR2RGB)


# Hien thi ket qua
plt.figure(figsize=(14, 8))

images = [
    (image_display, "Original"),
    (lab_display, "CLAHE - Lab"),
    (hsv_display, "CLAHE - HSV"),
    (rgb_display, "CLAHE - RGB")
]

for i, (img, title) in enumerate(images):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

print("Lab:", result_lab.shape)
print("HSV:", result_hsv.shape)
print("RGB:", result_rgb.shape)

print("KIEM TRA HOAN TAT!")