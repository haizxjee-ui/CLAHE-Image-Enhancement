import pandas as pd
import requests
from pathlib import Path
import cv2

# Đường dẫn dữ liệu
metadata_path = Path("datasets/sample_50/metadata.csv")
output_dir = Path("datasets/sample_50/images")

# Tạo thư mục lưu ảnh
output_dir.mkdir(parents=True, exist_ok=True)

# Đọc danh sách 50 ảnh
df = pd.read_csv(metadata_path)

# Địa chỉ nguồn ảnh
BASE_URL = (
    "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/"
    "images_1024x1024/"
)

success = 0
failed = []

print("===== BAT DAU TAI ANH =====")

for index, row in df.iterrows():

    filename = row["Image Index"]
    output_path = output_dir / filename

    # Bỏ qua nếu ảnh đã tồn tại và đọc được
    if output_path.exists() and cv2.imread(str(output_path)) is not None:
        success += 1
        print(f"[DA CO] {filename}")
        continue

    url = BASE_URL + filename

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Kiểm tra nội dung ảnh trước khi lưu
        image_array = bytearray(response.content)
        image = cv2.imdecode(
            __import__("numpy").frombuffer(image_array, dtype="uint8"),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise ValueError("Du lieu tai ve khong phai anh hop le")

        output_path.write_bytes(response.content)

        success += 1
        print(f"[OK] {filename}")

    except Exception as error:
        failed.append(filename)
        print(f"[LOI] {filename}: {error}")

print("\n===== KET QUA =====")
print("Tong so anh:", len(df))
print("Tai thanh cong:", success)
print("That bai:", len(failed))

if failed:
    print("\nDanh sach anh loi:")
    for filename in failed:
        print(filename)