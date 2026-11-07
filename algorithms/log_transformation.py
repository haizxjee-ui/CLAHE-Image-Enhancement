import numpy as np


def log_transformation(image, c=255 / np.log(256)):
    """
    Thuc hien Log Transformation cho anh grayscale 8-bit.
    """

    if image is None:
        raise ValueError("Anh dau vao khong hop le")

    if image.ndim != 2:
        raise ValueError("Anh phai la grayscale")

    if image.dtype != np.uint8:
        raise ValueError("Anh phai co kieu du lieu uint8")

    if c <= 0:
        raise ValueError("He so c phai lon hon 0")

    # Ap dung cong thuc bien doi logarit
    result = c * np.log1p(image.astype(np.float64))

    return np.clip(np.round(result), 0, 255).astype(np.uint8)