
import numpy as np
import time


def calculate_entropy(image):
    histogram = np.bincount(
        image.flatten(),
        minlength=256
    )

    probabilities = histogram / image.size
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return entropy


def calculate_rms_contrast(image):
    return np.std(image)


def calculate_ambe(original, processed):
    original_mean = np.mean(original)
    processed_mean = np.mean(processed)

    return abs(original_mean - processed_mean)


def calculate_processing_time(function, *args, **kwargs):
    start_time = time.perf_counter()

    result = function(*args, **kwargs)

    end_time = time.perf_counter()

    processing_time = (end_time - start_time) * 1000

    return result, processing_time
