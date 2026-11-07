import cv2
import matplotlib.pyplot as plt

from algorithms.gamma_correction import gamma_correction


# Doc anh X-quang
image_path = "datasets/sample_50/images/00026406_000.png"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Khong tim thay anh")


# Ap dung Gamma Correction
result = gamma_correction(image, gamma=1.5)


# Hien thi anh
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Anh goc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Gamma Correction (gamma=1.5)")
plt.axis("off")

plt.tight_layout()
plt.show()