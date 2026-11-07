import pandas as pd

# Đọc metadata
csv_path = "datasets/raw/Data_Entry_2017_v2020.csv"
df = pd.read_csv(csv_path)

# Chỉ lấy ảnh chụp tư thế PA
df_pa = df[df["View Position"] == "PA"].copy()

# Chia thành hai nhóm
df_normal = df_pa[
    df_pa["Finding Labels"] == "No Finding"
]

df_abnormal = df_pa[
    df_pa["Finding Labels"] != "No Finding"
]

# Đếm số bệnh nhân khác nhau
normal_patients = df_normal["Patient ID"].nunique()
abnormal_patients = df_abnormal["Patient ID"].nunique()

print("===== THONG KE ANH PA =====")

print("Tong so anh PA:", len(df_pa))

print("\nNhom No Finding:")
print("So anh:", len(df_normal))
print("So benh nhan:", normal_patients)

print("\nNhom Abnormal:")
print("So anh:", len(df_abnormal))
print("So benh nhan:", abnormal_patients)