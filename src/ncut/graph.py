# construct the graph representation of the image.
"""
this is where:
-pixels becomes nodes
-similarities becomes weighted edges
-sparse adjacency matrix is created

this handles the mathematical graph model.

W(i, j); we are building W = adjacency matrix and degree vector d + sparse graph structure

we'll have main functions as :
A) pixel idexing
    convert (y,x) <-> node index

B) weighted computation
    implements:
    Wij = exp(.....).exp(....) 

C) Sparse matrix construction
    build W
"""

#minimal implementation

import numpy as np
from scipy.sparse import coo_matrix

def pixel_index(y, x, width):
    return y * width + x

def compute_weight(intensity_a, intensity_b, spatial_dist, 
                   sigma_i=0.1, sigma_x=4.0):
    feature_term = np.exp(-((intensity_a - intensity_b) ** 2) / (sigma_i ** 2))

    spatial_term = np.exp(-(spatial_dist ** 2) / (sigma_x ** 2))

    return feature_term * spatial_term


def build_weight_matrix(image, radius=3):

    h, w = image.shape


    rows = []
    cols = []
    vals = []

    for y in range(h):
        for x in range(w):

            i = pixel_index(y, x, w)

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):

                    ny = y + dy
                    nx = x + dx

                    if nx < 0 or nx >= w:
                        continue
                    
                    if nx < 0 or nx >= w:
                        continue

                    if dx == 0 and dy == 0:
                        continue

                    j = pixel_index(ny, nx, w)

                    spatial_dist = np.sqrt(dx**2 + dy**2)

                    weight = compute_weight(
                        image[y, x],
                        image[ny,  nx],
                        spatial_dist
                    )

                    rows.append(i)
                    cols.append(j)
                    vals.append(weight)
    
    w = coo_matrix(
        (vals, (rows, cols)),
        shape=(h * w, h * w)
    )

    return w.tocsr()