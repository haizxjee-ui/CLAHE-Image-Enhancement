import cv2
import numpy as np
import matplotlib.pyplot as plt

from algorithms.histogram_equalization import histogram_equalization
from algorithms.adaptive_histogram_equalization import adaptive_histogram_equalization
from algorithms.clahe import clahe
from algorithms.gamma_correction import gamma_correction
from algorithms.log_transformation import log_transformation


# Doc anh X-quang
image_path = "datasets/sample_50/images/00026406_000.png"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Khong tim thay anh")


# Ap dung cac phuong phap
results = {
    "Original": image,
    "HE": histogram_equalization(image),
    "AHE": adaptive_histogram_equalization(image),
    "CLAHE": clahe(image, grid_size=(8, 8), clip_limit=2),
    "Gamma": gamma_correction(image, gamma=1.5),
    "Log": log_transformation(image)
}


# Hien thi anh va histogram
plt.figure(figsize=(15, 10))

for i, (name, result) in enumerate(results.items()):

    # Hien thi anh
    plt.subplot(6, 2, 2 * i + 1)
    plt.imshow(result, cmap="gray")
    plt.title(name)
    plt.axis("off")

    # Hien thi histogram
    plt.subplot(6, 2, 2 * i + 2)
    plt.hist(result.flatten(), bins=256, range=(0, 256))
    plt.title("Histogram - " + name)
    plt.xlim(0, 255)

plt.tight_layout()
plt.show()


# Tinh cac chi so
print("\nKET QUA SO SANH")
print("-" * 55)

for name, result in results.items():

    histogram = np.bincount(
        result.flatten(),
        minlength=256
    )

    probabilities = histogram / result.size
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    mean = np.mean(result)
    std = np.std(result)

    print(
        f"{name:10} | "
        f"Mean: {mean:7.2f} | "
        f"Std: {std:7.2f} | "
        f"Entropy: {entropy:.4f}"
    )