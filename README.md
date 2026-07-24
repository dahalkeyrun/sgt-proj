# Image Segmentation using Spectral Graph Theory

This is a 6th-semester Graph Theory mini-project implementing image segmentation
with the **Normalized Cuts** method from Shi and Malik (2000).

The project is currently in the prototype stage. The graph-construction and
image-loading pipelines are working, and a simple synthetic normalized-cuts
solver diagnostic is available.

## Project Structure

```text
sgt/
+-- README.md
+-- TASK.md
+-- requirements.txt
+-- data/
|   +-- input/
|       +-- pipeline_sample.png
+-- scripts/
|   +-- run_segmentation.py
+-- src/
|   +-- ncut/
|       +-- __init__.py
|       +-- graph.py
|       +-- ncut.py
|       +-- partition.py
|       +-- recursive.py
|       +-- utils.py
|       +-- visualize.py
+-- tests/
    +-- test_graph.py
    +-- test_image_pipeline.py
    +-- test_ncut.py
```

All reusable project code should live in `src/ncut/`. Scripts and tests should
import through the package, for example:

```python
from ncut.graph import build_graph_structures, build_weight_matrix
from ncut.utils import load_image
```

## Current Features

- Image loading from disk.
- Grayscale and RGB conversion.
- Image normalization to `[0, 1]`.
- Image resizing/downsampling for memory safety.
- Pixel graph construction from grayscale or color images.
- Gaussian pixel similarity weights using intensity and spatial distance.
- Explicit self-edge removal, so `W[i, i] = 0`.
- Strict radius boundary behavior: pixels connect only when `distance < radius`.
- Manual CSR-like graph arrays:
  - `values`
  - `col_indices`
  - `row_pointers`
  - `degree_vector`
- SciPy `csr_matrix` helper through `build_weight_matrix`.
- Prototype Fiedler-vector solver diagnostic.

## Normalized Cuts Overview

Normalized Cuts models an image as a graph:

- each pixel is a graph node
- nearby pixels are connected by weighted edges
- edge weights measure pixel similarity

The similarity weight is based on both intensity/color difference and spatial
distance. The goal is to split the graph into meaningful regions while avoiding
unbalanced cuts.

The core eigenproblem is:

```text
(D - W)y = lambda D y
```

where:

- `W` is the pixel similarity matrix
- `D` is the degree matrix
- `D - W` is the graph Laplacian
- the second smallest eigenvector is used for bipartitioning

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Optional GPU support:

- This repository is designed to support CPU and GPU backends.
- The current implementation supports `backend="cpu"` by default.
- To enable GPU execution in future work, install a CUDA-enabled CuPy package, for example:

```powershell
pip install cupy-cuda12x
```

Current pinned dependencies:

```text
numpy==1.26.4
scipy==1.11.3
pillow==11.1.0
matplotlib==3.10.1
```

## Diagnostics

Run the graph-construction verification:

```powershell
python tests\test_graph.py
```

This checks:

- graph size
- input validation
- self-edge removal
- SciPy CSR matrix shape
- zero diagonal
- degree vector consistency

Run the image-loading pipeline verification:

```powershell
python tests\test_image_pipeline.py
```

This expects:

```text
data/input/pipeline_sample.png
```

Run the current solver diagnostic:

```powershell
python tests\test_ncut.py
```

## Current Limitations

- `partition.py`, `recursive.py`, and `visualize.py` are not implemented yet.
- The solver is still a prototype and should be upgraded to use SciPy sparse
  eigen-solvers.
- The command-line segmentation script is not integrated yet.
- Large images are expensive because normalized cuts builds a graph over pixels.
- Parameter tuning for `sigma_i`, `sigma_x`, and `radius` is still needed.

## Next Milestone

The next milestone is to implement binary partitioning:

```text
image -> graph weights -> Fiedler vector -> binary mask
```

After that, the project can add visualization overlays and recursive splitting.
