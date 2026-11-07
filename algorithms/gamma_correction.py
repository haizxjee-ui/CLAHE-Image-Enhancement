import numpy as np


def gamma_correction(image, gamma=1.5):
    """
    Thuc hien Gamma Correction cho anh grayscale 8-bit.
    """

    if image is None:
        raise ValueError("Anh dau vao khong hop le")

    if image.ndim != 2:
        raise ValueError("Anh phai la grayscale")

    if image.dtype != np.uint8:
        raise ValueError("Anh phai co kieu du lieu uint8")

    if gamma <= 0:
        raise ValueError("Gamma phai lon hon 0")

    # Chuan hoa gia tri pixel ve khoang 0-1
    normalized = image / 255.0

    # Ap dung cong thuc Gamma Correction
    corrected = np.power(normalized, gamma)

    # Dua gia tri pixel ve khoang 0-255
    result = np.round(corrected * 255)

    return np.clip(result, 0, 255).astype(np.uint8)