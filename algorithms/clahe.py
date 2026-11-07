
import numpy as np


def clip_histogram(histogram, clip_limit):
    """Gioi han histogram va phan phoi lai pixel du."""

    histogram = np.asarray(histogram, dtype=np.int64)

    if histogram.shape != (256,):
        raise ValueError("Histogram phai co 256 phan tu")

    if np.any(histogram < 0):
        raise ValueError("Histogram khong duoc co gia tri am")

    if clip_limit < 1:
        raise ValueError("Clip limit phai >= 1")

    clipped_histogram = np.minimum(histogram, clip_limit)

    excess = np.sum(histogram - clipped_histogram)

    batch = excess // 256
    residual = excess % 256

    clipped_histogram += batch

    if residual > 0:
        step = max(256 // residual, 1)
        indices = np.arange(0, 256, step)[:residual]
        clipped_histogram[indices] += 1

    return clipped_histogram


def create_clahe_lut(histogram):
    """Tao bang tra cuu LUT tu histogram da clipping."""

    cdf = np.cumsum(histogram)
    total_pixels = cdf[-1]

    if total_pixels == 0:
        raise ValueError("Histogram khong co pixel")

    lut = np.round(cdf * 255 / total_pixels)
    lut = np.clip(lut, 0, 255)

    return lut.astype(np.uint8)


def create_tile_luts(image, grid_size=(8, 8), clip_limit=2.0):
    """Tao LUT cho tung tile cua anh."""

    if image is None or image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Anh phai la grayscale uint8")

    height, width = image.shape
    grid_rows, grid_cols = grid_size

    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("So tile phai lon hon 0")

    if grid_rows > height or grid_cols > width:
        raise ValueError("So tile vuot qua kich thuoc anh")

    if clip_limit <= 0:
        raise ValueError("Clip limit phai lon hon 0")

    row_blocks = np.array_split(image, grid_rows, axis=0)

    lut_grid = np.zeros(
        (grid_rows, grid_cols, 256),
        dtype=np.uint8
    )

    for row_idx, row_block in enumerate(row_blocks):
        col_blocks = np.array_split(row_block, grid_cols, axis=1)

        for col_idx, tile in enumerate(col_blocks):
            histogram = np.bincount(
                tile.flatten(),
                minlength=256
            )

            actual_clip_limit = max(
                1,
                int(clip_limit * tile.size / 256)
            )

            clipped_histogram = clip_histogram(
                histogram,
                actual_clip_limit
            )

            lut_grid[row_idx, col_idx] = create_clahe_lut(
                clipped_histogram
            )

    return lut_grid


def apply_clahe_interpolation(image, lut_grid):
    """Ap dung noi suy bilinear giua cac tile."""

    height, width = image.shape
    grid_rows, grid_cols = lut_grid.shape[:2]

    row_centers = np.array([
        block.mean()
        for block in np.array_split(np.arange(height), grid_rows)
    ])

    col_centers = np.array([
        block.mean()
        for block in np.array_split(np.arange(width), grid_cols)
    ])

    y = np.arange(height)
    x = np.arange(width)

    y1 = np.clip(
        np.searchsorted(row_centers, y) - 1,
        0, grid_rows - 1
    )
    y2 = np.minimum(y1 + 1, grid_rows - 1)

    x1 = np.clip(
        np.searchsorted(col_centers, x) - 1,
        0, grid_cols - 1
    )
    x2 = np.minimum(x1 + 1, grid_cols - 1)

    v = np.zeros(height)
    u = np.zeros(width)

    mask_y = y2 != y1
    mask_x = x2 != x1

    v[mask_y] = (
        (y[mask_y] - row_centers[y1[mask_y]]) /
        (row_centers[y2[mask_y]] - row_centers[y1[mask_y]])
    )

    u[mask_x] = (
        (x[mask_x] - col_centers[x1[mask_x]]) /
        (col_centers[x2[mask_x]] - col_centers[x1[mask_x]])
    )

    v = np.clip(v, 0, 1)[:, None]
    u = np.clip(u, 0, 1)[None, :]

    q11 = lut_grid[y1[:, None], x1[None, :], image]
    q21 = lut_grid[y1[:, None], x2[None, :], image]
    q12 = lut_grid[y2[:, None], x1[None, :], image]
    q22 = lut_grid[y2[:, None], x2[None, :], image]

    result = (
        q11 * (1 - u) * (1 - v) +
        q21 * u * (1 - v) +
        q12 * (1 - u) * v +
        q22 * u * v
    )

    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def clahe(image, grid_size=(8, 8), clip_limit=2.0):
    """Thuat toan CLAHE tu cai dat bang NumPy."""

    if image is None or image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Anh phai la grayscale uint8")

    lut_grid = create_tile_luts(
        image,
        grid_size,
        clip_limit
    )

    result = apply_clahe_interpolation(image, lut_grid)

    return result
