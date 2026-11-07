
import numpy as np

from algorithms.histogram_equalization import histogram_equalization


def adaptive_histogram_equalization(image, grid_size=(8, 8)):
    """
    Thuc hien Adaptive Histogram Equalization (AHE).

    Args:
        image: Anh grayscale 8-bit.
        grid_size: So block theo (hang, cot).

    Returns:
        Anh sau khi xu ly AHE.
    """

    # Buoc 1: Kiem tra anh dau vao
    if image is None:
        raise ValueError("Anh dau vao khong hop le")

    if image.ndim != 2:
        raise ValueError("Anh phai la grayscale")

    if image.dtype != np.uint8:
        raise ValueError("Anh phai co kieu du lieu uint8")

    # Buoc 2: Lay kich thuoc anh va luoi block
    height, width = image.shape
    grid_rows, grid_cols = grid_size

    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("So block phai lon hon 0")

    if grid_rows > height or grid_cols > width:
        raise ValueError("So block vuot qua kich thuoc anh")

    # Buoc 3: Chia anh thanh cac block theo hang
    row_blocks = np.array_split(image, grid_rows, axis=0)

    # Tao anh ket qua
    result = np.zeros_like(image)

    current_row = 0

    # Buoc 4: Xu ly tung block
    for row_block in row_blocks:

        # Chia moi hang thanh cac block theo cot
        col_blocks = np.array_split(
            row_block,
            grid_cols,
            axis=1
        )

        current_col = 0

        for block in col_blocks:

            # Can bang histogram rieng cho block
            equalized_block = histogram_equalization(block)

            # Lay kich thuoc block
            block_height, block_width = block.shape

            # Ghep block vao anh ket qua
            result[
                current_row:current_row + block_height,
                current_col:current_col + block_width
            ] = equalized_block

            current_col += block_width

        current_row += row_block.shape[0]

    return result
