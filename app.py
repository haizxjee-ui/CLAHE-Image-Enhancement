
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from algorithms.histogram_equalization import histogram_equalization
from algorithms.adaptive_histogram_equalization import adaptive_histogram_equalization
from algorithms.clahe import clahe
from algorithms.gamma_correction import gamma_correction
from algorithms.log_transformation import log_transformation

from algorithms.color_processing import (
    clahe_lab,
    clahe_hsv,
    clahe_rgb
)

from utils.image_metrics import (
    calculate_entropy,
    calculate_rms_contrast,
    calculate_ambe,
    calculate_processing_time
)


# Cau hinh trang
st.set_page_config(
    page_title="CLAHE X-ray Enhancement",
    page_icon="🩻",
    layout="wide"
)

st.title("Ứng dụng tăng cường ảnh")
st.write("So sánh các phương pháp xử lý và tăng cường độ tương phản ảnh.")


# Tai anh len
uploaded_file = st.file_uploader(
    "Chọn ảnh",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    # Doc anh mau
    file_bytes = np.frombuffer(
        uploaded_file.read(),
        dtype=np.uint8
    )

    color_image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if color_image is None:
        st.error("Không thể đọc ảnh!")
        st.stop()

    # Chuyen anh mau sang grayscale
    image = cv2.cvtColor(
        color_image,
        cv2.COLOR_BGR2GRAY
    )

    st.sidebar.header("Cài đặt xử lý ảnh")

    # Chon phuong phap
    method = st.sidebar.selectbox(
        "Chọn phương pháp",
        [
            "HE",
            "AHE",
            "CLAHE",
            "CLAHE - Lab",
            "CLAHE - HSV",
            "CLAHE - RGB",
            "Gamma Correction",
            "Log Transformation"
        ]
    )

    # Thiet lap tham so
    if method == "AHE":
        grid_size = st.sidebar.slider(
            "Số block",
            2, 16, 8
        )

    elif method in [
        "CLAHE",
        "CLAHE - Lab",
        "CLAHE - HSV",
        "CLAHE - RGB"
    ]:
        grid_size = st.sidebar.slider(
            "Số tile",
            2, 16, 8
        )

        clip_limit = st.sidebar.slider(
            "Clip Limit",
            1.0, 5.0, 2.0, 0.5
        )

    elif method == "Gamma Correction":
        gamma = st.sidebar.slider(
            "Gamma",
            0.1, 3.0, 1.5, 0.1
        )

    # Xu ly anh va do thoi gian
    if method == "HE":
        result, processing_time = calculate_processing_time(
            histogram_equalization,
            image
        )

    elif method == "AHE":
        result, processing_time = calculate_processing_time(
            adaptive_histogram_equalization,
            image,
            grid_size=(grid_size, grid_size)
        )

    elif method == "CLAHE":
        result, processing_time = calculate_processing_time(
            clahe,
            image,
            grid_size=(grid_size, grid_size),
            clip_limit=clip_limit
        )

    elif method == "CLAHE - Lab":
        result, processing_time = calculate_processing_time(
            clahe_lab,
            color_image,
            grid_size=(grid_size, grid_size),
            clip_limit=clip_limit
        )

    elif method == "CLAHE - HSV":
        result, processing_time = calculate_processing_time(
            clahe_hsv,
            color_image,
            grid_size=(grid_size, grid_size),
            clip_limit=clip_limit
        )

    elif method == "CLAHE - RGB":
        result, processing_time = calculate_processing_time(
            clahe_rgb,
            color_image,
            grid_size=(grid_size, grid_size),
            clip_limit=clip_limit
        )

    elif method == "Gamma Correction":
        result, processing_time = calculate_processing_time(
            gamma_correction,
            image,
            gamma=gamma
        )

    else:
        result, processing_time = calculate_processing_time(
            log_transformation,
            image
        )

    # Chuan bi anh hien thi va anh grayscale de danh gia
    if method in [
        "CLAHE - Lab",
        "CLAHE - HSV",
        "CLAHE - RGB"
    ]:

        display_result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )

        gray_result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2GRAY
        )

    else:
        display_result = result
        gray_result = result

    gray_original = image

    # Hien thi anh goc va anh ket qua
    st.subheader("So sánh ảnh trước và sau xử lý")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            cv2.cvtColor(
                color_image,
                cv2.COLOR_BGR2RGB
            ),
            caption="Ảnh gốc",
            width="stretch"
        )

    with col2:
        st.image(
            display_result,
            caption=method,
            width="stretch",
            clamp=True
        )

    # Hien thi Histogram
    st.subheader("So sánh Histogram")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4)
    )

    axes[0].hist(
        gray_original.flatten(),
        bins=256,
        range=(0, 256)
    )

    axes[0].set_title("Histogram ảnh gốc")
    axes[0].set_xlim(0, 255)

    axes[1].hist(
        gray_result.flatten(),
        bins=256,
        range=(0, 256)
    )

    axes[1].set_title("Histogram ảnh sau xử lý")
    axes[1].set_xlim(0, 255)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Tinh cac chi so danh gia
    entropy = calculate_entropy(gray_result)

    rms_contrast = calculate_rms_contrast(gray_result)

    ambe = calculate_ambe(
        gray_original,
        gray_result
    )

    # Hien thi cac chi so
    st.subheader("Các chỉ số đánh giá")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Entropy",
        f"{entropy:.4f}"
    )

    col2.metric(
        "RMS Contrast",
        f"{rms_contrast:.4f}"
    )

    col3.metric(
        "AMBE",
        f"{ambe:.4f}"
    )

    col4.metric(
        "Processing Time",
        f"{processing_time:.2f} ms"
    )

else:
    st.info("Vui lòng tải ảnh để bắt đầu xử lý.")
