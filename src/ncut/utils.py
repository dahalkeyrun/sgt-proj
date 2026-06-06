from pathlib import Path

import numpy as np
import PIL.Image as pil

def load_image(image_path, mode="grayscale", max_size=128):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if mode not in ("grayscale", "rgb"):
        raise ValueError("mode must be either in grayscale or rgb")
    
    image = pil.open(image_path)

    if mode == "grayscale":
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = resize_image(image, max_size=max_size)

    image_array = np.asarray(image, dtype=np.float64)
    image_array /= 255.0

    return image_array

def resize_image(image, max_size=128):

    if max_size is None:
        return image
    
    if max_size <= 0:
        raise ValueError("max_size must be positive or None")
    
    width, height = image.size
    largest_side = max(width, height)

    if largest_side <= max_size:
        return image
    
    scale = max_size / largest_side
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    return image.resize((new_width, new_height), pil.Resampling.LANCZOS)


def flatten_image(image):
    if image.ndim == 2:
        return image.reshape(-1)
    
    if image.ndim == 3:
        height, width, channels = image.shape
        return image.reshape(height * width, channels)
    
    raise ValueError("image must be either 2D grayscale or 3D RGB")


def reshape_vector_to_image(vector, height, width):

    vector = np.asarray(vector)

    expected_size = height * width
    if vector.size != expected_size:
        raise ValueError(
            f"Cannot reshape vector of size {vector.size} into image shape "
            f"({height}, {width})"
        )
    
    return vector.reshape((height, width))