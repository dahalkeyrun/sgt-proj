import math
import numpy as np

def calculate_spatial_distance_sq(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

def calculate_intensity_distance_sq(p1, p2):
    diff = p1 - p2
    return np.sum(diff ** 2)

def build_graph_structures(image, sigma_i, sigma_x, radius):

    height, width = image.shape[:2]
    num_pixels = height * width

    two_sigma_i_sq = 2.0 * (sigma_i ** 2)
    two_sigma_x_sq = 2.0 * (sigma_x ** 2)

    radius_sq = radius ** 2


    values = []
    col_indices = []
    row_pointers =[0]
    degree_vector = np.zeros(num_pixels, dtype=np.float64)

    for y1 in range(height):
        for x1 in range(width):

            i = y1 * width + x1
            pixel_i_intensity = image[y1, x1]

            y_min = max(0, y1 - radius)
            y_max = min(height, y1 + radius + 1)
            x_min = max(0, x1 - radius)
            x_max = min(width, x1 + radius + 1)

            running_degree_sum = 0.0

            for y2 in range(y_min, y_max):
                for x2 in range(x_min, x_max):

                    j = y2 * width + x2

                    dist_x_sq = calculate_spatial_distance_sq(x1, y1, x2, y2)
                    if dist_x_sq >= radius_sq:
                        continue

                    pixel_j_intensity = image[y2, x2]
                    dist_i_sq = calculate_intensity_distance_sq(pixel_i_intensity, pixel_i_intensity)

                    weight = math.exp(-dist_i_sq / two_sigma_i_sq) * math.exp(-dist_x_sq / two_sigma_x_sq)

                    values.append(weight)
                    col_indices.append(j)
                    running_degree_sum += weight

            degree_vector[i] = running_degree_sum

            row_pointers.append(len(values))

    return(
            np.array(values, dtype=np.float64),
            np.array(col_indices, dtype=np.int32),
            np.array(row_pointers, dtype=np.int32),
            degree_vector
            )
