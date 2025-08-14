
import numpy as np
from typing import Tuple, Iterator

def image_pyramid(image, scale=1.5, min_size=(64, 128)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        if w < min_size[0] or h < min_size[1]:
            break
        image = _resize_cv(image, (w, h))
        yield image

def _resize_cv(img, size_wh):
    # Lazy import to avoid hard dependency for unit tests
    import cv2
    return cv2.resize(img, size_wh, interpolation=cv2.INTER_AREA)

def sliding_window(image, stepSize: int, windowSize: Tuple[int, int]) -> Iterator[Tuple[int, int, np.ndarray]]:
    for y in range(0, image.shape[0] - windowSize[1] + 1, stepSize):
        for x in range(0, image.shape[1] - windowSize[0] + 1, stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
