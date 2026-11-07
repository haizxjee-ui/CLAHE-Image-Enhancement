import cv2

from algorithms.clahe import clahe


def validate_color_image(image):
    if image is None:
        raise ValueError("Anh dau vao khong hop le")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Anh phai co 3 kenh mau")

    if image.dtype != "uint8":
        raise ValueError("Anh phai co kieu du lieu uint8")


# 1. CLAHE tren kenh L cua Lab
def clahe_lab(image, grid_size=(8, 8), clip_limit=2.0):
    validate_color_image(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    l_result = clahe(
        l_channel,
        grid_size,
        clip_limit
    )

    result_lab = cv2.merge([
        l_result,
        a_channel,
        b_channel
    ])

    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


# 2. CLAHE tren kenh V cua HSV
def clahe_hsv(image, grid_size=(8, 8), clip_limit=2.0):
    validate_color_image(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_channel, s_channel, v_channel = cv2.split(hsv)

    v_result = clahe(
        v_channel,
        grid_size,
        clip_limit
    )

    result_hsv = cv2.merge([
        h_channel,
        s_channel,
        v_result
    ])

    return cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)


# 3. CLAHE doc lap tren tung kenh RGB
def clahe_rgb(image, grid_size=(8, 8), clip_limit=2.0):
    validate_color_image(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    r_channel, g_channel, b_channel = cv2.split(rgb)

    r_result = clahe(r_channel, grid_size, clip_limit)
    g_result = clahe(g_channel, grid_size, clip_limit)
    b_result = clahe(b_channel, grid_size, clip_limit)

    result_rgb = cv2.merge([
        r_result,
        g_result,
        b_result
    ])

    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)