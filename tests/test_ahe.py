
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from algorithms.adaptive_histogram_equalization import (
    adaptive_histogram_equalization
)

# Doc anh X-quang
image_path = Path(
    "datasets/sample_50/images/00026406_000.png"
)

image = cv2.imread(
    str(image_path),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise ValueError("Khong doc duoc anh")

# Thuc hien AHE voi luoi 8x8
result = adaptive_histogram_equalization(
    image,
    grid_size=(8, 8)
)

# Hien thi ket qua
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.hist(image.ravel(), bins=256, range=(0, 256))
plt.title("Original Histogram")

plt.subplot(2, 2, 3)
plt.imshow(result, cmap="gray")
plt.title("AHE - 8x8")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.hist(result.ravel(), bins=256, range=(0, 256))
plt.title("AHE Histogram")

plt.tight_layout()
plt.show()
