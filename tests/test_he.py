
import cv2
import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path
from algorithms.histogram_equalization import histogram_equalization


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

# Ap dung HE tu cai dat
result = histogram_equalization(image)
# So sanh voi OpenCV
opencv_result = cv2.equalizeHist(image)

# Tinh sai so tuyet doi
difference = cv2.absdiff(result, opencv_result)

print("===== KIEM TRA HE =====")
print("Kich thuoc anh:", result.shape)
print("Sai so lon nhat:", difference.max())
print("Sai so trung binh:", difference.mean())

if np.array_equal(result, opencv_result):
    print("KET QUA: Hai thuat toan cho ket qua giong nhau")
else:
    print("KET QUA: Hai thuat toan co su khac biet")

# Hien thi anh va histogram
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
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.hist(result.ravel(), bins=256, range=(0, 256))
plt.title("Equalized Histogram")

plt.tight_layout()
plt.show()
