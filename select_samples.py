import pandas as pd
from pathlib import Path

# 1. Doc metadata
csv_path = "datasets/raw/Data_Entry_2017_v2020.csv"
df = pd.read_csv(csv_path)

# 2. Chi lay anh PA
df_pa = df[df["View Position"] == "PA"].copy()

# 3. Chia hai nhom
normal = df_pa[df_pa["Finding Labels"] == "No Finding"]
abnormal = df_pa[df_pa["Finding Labels"] != "No Finding"]

# 4. Loai bo benh nhan xuat hien o ca hai nhom
normal_ids = set(normal["Patient ID"])
abnormal_ids = set(abnormal["Patient ID"])

normal_only_ids = normal_ids - abnormal_ids

normal = normal[normal["Patient ID"].isin(normal_only_ids)]

# 5. Moi benh nhan chi lay mot anh
normal = normal.groupby("Patient ID").sample(
    n=1, random_state=42
)

abnormal = abnormal.groupby("Patient ID").sample(
    n=1, random_state=42
)

# 6. Chon 25 anh moi nhom
normal = normal.sample(n=25, random_state=42).copy()
abnormal = abnormal.sample(n=25, random_state=42).copy()

normal["Group"] = "Normal"
abnormal["Group"] = "Abnormal"

# 7. Gop du lieu
selected = pd.concat([normal, abnormal], ignore_index=True)

selected = selected.sample(
    frac=1, random_state=42
).reset_index(drop=True)

# 8. Luu ket qua
output_dir = Path("datasets/sample_50")
output_dir.mkdir(parents=True, exist_ok=True)

selected.to_csv(
    output_dir / "metadata.csv",
    index=False
)

# 9. In ket qua
print("===== KET QUA CHON ANH =====")
print("Tong so anh:", len(selected))
print("\nSo luong theo nhom:")
print(selected["Group"].value_counts())

print("\nSo benh nhan duy nhat:",
      selected["Patient ID"].nunique())

print("\nDa luu metadata thanh cong!")

print("\n===== DANH SACH 10 ANH DA CHON =====")

print(
    selected[
        ["Image Index", "Patient ID", "Finding Labels", "Group"]
    ].head(10).to_string(index=False)
)