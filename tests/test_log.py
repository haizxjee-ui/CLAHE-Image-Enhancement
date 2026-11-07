import cv2
import matplotlib.pyplot as plt

from algorithms.log_transformation import log_transformation


# Doc anh X-quang
image_path = "datasets/sample_50/images/00026406_000.png"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Khong tim thay anh")


# Ap dung Log Transformation
result = log_transformation(image)


# Hien thi anh
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Anh goc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Log Transformation")
plt.axis("off")

plt.tight_layout()
plt.show()