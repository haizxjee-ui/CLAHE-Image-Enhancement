
import numpy as np


def histogram_equalization(image):
    """
    Thuc hien Histogram Equalization cho anh grayscale 8-bit.

    Args:
        image: Anh grayscale co gia tri pixel tu 0 den 255.

    Returns:
        Anh sau khi can bang histogram.
    """

    # Buoc 1: Kiem tra anh dau vao
    if image is None:
        raise ValueError("Anh dau vao khong hop le")

    if image.ndim != 2:
        raise ValueError("Anh dau vao phai la anh grayscale")

    if image.dtype != np.uint8:
        raise ValueError("Anh phai co kieu du lieu uint8")

    # Buoc 2: Tinh histogram
    histogram = np.bincount(
        image.flatten(),
        minlength=256
    )

    # Buoc 3: Tinh CDF
    cdf = np.cumsum(histogram)

    # Buoc 4: Tim CDF nho nhat khac 0
    cdf_min = cdf[cdf > 0].min()

    # Buoc 5: Tinh tong so pixel
    total_pixels = image.size

    # Buoc 6: Tinh bang tra cuu muc xam moi
    if total_pixels == cdf_min:
        return image.copy()

    lookup_table = np.round(
        (cdf - cdf_min) /
        (total_pixels - cdf_min) * 255
    )

    lookup_table = np.clip(
        lookup_table, 0, 255
    ).astype(np.uint8)

    # Buoc 7: Thay the gia tri pixel
    equalized_image = lookup_table[image]

    return equalized_image
