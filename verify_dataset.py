import cv2
import pandas as pd

from pathlib import Path

# Đường dẫn dữ liệu
metadata_path = Path("datasets/sample_50/metadata.csv")
image_dir = Path("datasets/sample_50/images")

# Đọc danh sách ảnh
df = pd.read_csv(metadata_path)

valid_images = 0
invalid_images = []
image_sizes = set()

print("===== KIEM TRA DU LIEU ANH =====")

for filename in df["Image Index"]:
    image_path = image_dir / filename

    # Đọc ảnh
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        invalid_images.append(filename)
        print(f"[LOI] {filename}")
        continue

    valid_images += 1
    height, width = image.shape[:2]
    image_sizes.add((width, height))

    print(f"[OK] {filename} - {width}x{height}")

print("\n===== TONG KET =====")
print("So anh trong metadata:", len(df))
print("So anh doc duoc:", valid_images)
print("So anh loi:", len(invalid_images))
print("Kich thuoc anh:", image_sizes)

print("\n===== PHAN LOAI =====")
print(df["Group"].value_counts())