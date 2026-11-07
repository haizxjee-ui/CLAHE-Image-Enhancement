
import cv2
import csv
import os
import numpy as np

from algorithms.clahe import clahe


# Doc anh X-quang
image_path = "datasets/sample_50/images/00026406_000.png"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Khong tim thay anh")


# Cac tham so khao sat
grid_sizes = [4, 8, 16]
clip_limits = [1, 2, 3, 4, 5]


# Tao thu muc luu ket qua
output_folder = "results/parameter_study"
os.makedirs(output_folder, exist_ok=True)


# Tao file luu thong tin
csv_path = os.path.join(output_folder, "results.csv")

with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Grid Size",
        "Clip Limit",
        "Mean",
        "Standard Deviation",
        "Entropy"
    ])

    # Chay 15 cau hinh
    for grid_size in grid_sizes:

        for clip_limit in clip_limits:

            result = clahe(
                image,
                grid_size=(grid_size, grid_size),
                clip_limit=clip_limit
            )

            # Tinh histogram
            histogram = np.bincount(
                result.flatten(),
                minlength=256
            )

            probabilities = histogram / result.size
            probabilities = probabilities[probabilities > 0]

            entropy = -np.sum(
                probabilities * np.log2(probabilities)
            )

            # Tinh cac chi so
            mean = np.mean(result)
            std = np.std(result)

            # Luu anh
            filename = f"CLAHE_{grid_size}x{grid_size}_clip{clip_limit}.png"

            cv2.imwrite(
                os.path.join(output_folder, filename),
                result
            )

            # Luu thong tin vao CSV
            writer.writerow([
                f"{grid_size}x{grid_size}",
                clip_limit,
                round(mean, 2),
                round(std, 2),
                round(entropy, 4)
            ])

            print(
                f"Grid: {grid_size}x{grid_size} | "
                f"Clip: {clip_limit} | "
                f"Entropy: {entropy:.4f}"
            )

print("\nDA HOAN THANH 15 CAU HINH!")
print("Ket qua duoc luu tai:", output_folder)
