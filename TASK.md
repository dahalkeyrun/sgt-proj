# Project Tasks: Sparse Spectral Image Segmentation

This document tracks progress for the 6th-semester Graph Theory mini-project:
**Image Segmentation using Normalized Cuts**, based on Shi and Malik (2000).

The current repository is organized as a small Python package under `src/ncut`.
The first milestone is to produce a working bipartition segmentation pipeline:

```text
image -> graph weights -> eigenvector -> binary partition -> visualization
```

---

## Current Project Structure

```text
sgt/
+-- README.md
+-- TASK.md
+-- requirements.txt
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

## Package Layout and Import Rules

The `src/ncut/` directory is the single source of truth for project code. Keep
all algorithm modules inside this package so imports stay clean across tests,
scripts, and future notebooks.

- [x] Maintain the explicit package path `src/ncut/`.
- [x] Keep `src/ncut/__init__.py` so Python recognizes `ncut` as a package.
- [ ] Export the main public functions from `src/ncut/__init__.py` once module APIs stabilize.
- [ ] Use absolute imports from outside the package:

```python
from ncut.graph import build_graph_structures
from ncut.ncut import find_fiedler_vector
```

- [ ] Use package-relative imports inside `src/ncut` modules:

```python
from .graph import build_graph_structures
from .partition import partition_fiedler_vector
```

- [ ] Avoid creating duplicate top-level modules such as `engine.py`, `solver.py`,
  `logic.py`, or `preprocess.py` outside `src/ncut`.
- [ ] Avoid long-term `sys.path.append(...)` usage in project scripts and tests.
  Prefer editable installation or setting `PYTHONPATH=src` while developing.
- [ ] Keep scripts in `scripts/` as orchestration only; reusable logic belongs in
  `src/ncut/`.

---

## What Is Done

- [x] Repository structure created.
- [x] Basic `src/ncut` package created.
- [x] Graph construction started in `src/ncut/graph.py`.
- [x] Pixel-to-node mapping implemented.
- [x] Spatial distance calculation implemented.
- [x] Intensity/color distance calculation implemented.
- [x] Gaussian pixel similarity weight formula implemented.
- [x] Manual CSR-like graph storage implemented:
  - `values`
  - `col_indices`
  - `row_pointers`
  - `degree_vector`
- [x] Basic iterative solver started in `src/ncut/ncut.py`.
- [x] Simple diagnostic graph test added in `tests/test_graph.py`.
- [x] Image loading pipeline diagnostic added in `tests/test_image_pipeline.py`.
- [x] Simple synthetic left/right segmentation diagnostic added in `tests/test_ncut.py`.

---

## Immediate Fixes Needed

- [x] Fix `requirements.txt`.
  - Current environment versions are pinned for `numpy`, `scipy`, `pillow`, and `matplotlib`.
- [ ] Convert diagnostic test files into proper pytest tests.
- [ ] Remove temporary `sys.path.append(...)` imports from tests/scripts after
  package import setup is finalized.
- [ ] Fix the zero-degree branch in `d_inverse_w_product` inside `src/ncut/ncut.py`.
- [x] Decide whether self-edges should be included in the graph.
  - Self-edges are omitted so `W[i, i] = 0`.
- [x] Add input validation for:
  - image shape
  - positive `sigma_i`
  - positive `sigma_x`
  - positive `radius`
- [x] Update `README.md` to match the actual package structure.

---

## Core Implementation Plan

### 1. Graph Construction: `src/ncut/graph.py`

- [x] Build graph from image pixels.
- [x] Compute Gaussian weights using intensity and spatial distance.
- [x] Store weights in sparse CSR-like arrays.
- [x] Build degree vector.
- [x] Add parameter validation.
- [x] Remove or explicitly handle self-edges.
- [x] Document strict radius-boundary behavior.
- [x] Add helper to convert manual CSR arrays into `scipy.sparse.csr_matrix`.
- [x] Add support notes for grayscale and RGB images.
- [x] Add tests for small grayscale image.
- [x] Add tests for CSR matrix shape, diagonal, and degree row sums.

### 2. Eigen Solver: `src/ncut/ncut.py`

- [x] Implement basic `D^-1 W` multiplication.
- [x] Implement prototype Fiedler vector iteration.
- [x] Verify solver on simple synthetic left/right image.
- [ ] Fix zero-degree node handling.
- [ ] Replace or supplement manual solver with `scipy.sparse.linalg.eigsh`.
- [ ] Solve the normalized cuts generalized eigenproblem:

```text
(D - W)y = lambda D y
```

- [ ] Return both eigenvalue and eigenvector.
- [ ] Add convergence/error reporting.
- [ ] Add tests for solver shape and stable partition direction.

### 3. Partitioning: `src/ncut/partition.py`

- [ ] Implement thresholding by zero crossing.
- [ ] Implement thresholding by median value.
- [ ] Implement optional threshold search over the Fiedler vector.
- [ ] Implement NCut score:

```text
Ncut(A, B) = cut(A, B) / assoc(A, V) + cut(A, B) / assoc(B, V)
```

- [ ] Return binary mask, threshold, and NCut score.
- [ ] Add tests for mask shape and split quality.

### 4. Recursive Segmentation: `src/ncut/recursive.py`

- [ ] Implement single bipartition wrapper.
- [ ] Add recursive splitting.
- [ ] Add stopping conditions:
  - maximum depth
  - minimum segment size
  - maximum allowed NCut score
  - minimum eigenvector stability
- [ ] Return final 2D label map.
- [ ] Add tests for simple multi-region mock image.

### 5. Utilities and Preprocessing: `src/ncut/utils.py`

- [x] Load image from disk.
- [x] Convert image to grayscale or RGB.
- [x] Normalize pixel values to `[0, 1]`.
- [x] Resize/downsample large images for memory safety.
- [x] Reshape 1D vectors back to image grid.
- [ ] Create output directory helpers.

### 6. Visualization: `src/ncut/visualize.py`

- [ ] Plot original image.
- [ ] Plot Fiedler vector heatmap.
- [ ] Plot binary segmentation mask.
- [ ] Plot segmentation overlay on original image.
- [ ] Save result images to an output folder.

### 7. Command-Line Script: `scripts/run_segmentation.py`

- [ ] Accept image path from command line.
- [ ] Import reusable logic through the `ncut` package, not by duplicating code.
- [ ] Accept algorithm parameters:
  - `sigma_i`
  - `sigma_x`
  - `radius`
  - threshold method
  - optional max recursion depth
- [ ] Run the full segmentation pipeline.
- [ ] Save visualization outputs.
- [ ] Print useful summary information:
  - image size
  - graph node count
  - edge count
  - selected threshold
  - NCut score

---

## Testing Plan

- [ ] Install and configure `pytest`.
- [ ] Configure tests so `from ncut...` imports work consistently.
- [ ] Convert current diagnostic scripts into pytest functions.
- [x] Add graph construction tests.
- [ ] Add solver tests.
- [ ] Add partitioning tests.
- [x] Add preprocessing tests.
- [ ] Add one full pipeline integration test using a tiny mock image.

---

## Documentation Plan

- [x] Update `README.md` with the actual project structure.
- [x] Add short explanation of Normalized Cuts.
- [x] Add explanation of graph construction.
- [x] Add explanation of Fiedler vector partitioning.
- [x] Add setup instructions.
- [x] Add run instructions.
- [ ] Add example output section once visualization is complete.
- [x] Add limitations:
  - high memory usage for large images
  - parameter sensitivity
  - recursive segmentation needs tuning

---

## Suggested Work Order

1. Fix small solver bug in `src/ncut/ncut.py`.
2. Implement `src/ncut/partition.py`.
3. Implement visualization in `src/ncut/visualize.py`.
4. Fill `scripts/run_segmentation.py`.
5. Upgrade solver to use SciPy sparse eigen-solvers.
6. Convert current diagnostics into real pytest tests.
7. Add recursive segmentation in `src/ncut/recursive.py`.

---

## Team Task Split

### Kiran: Partitioning and Tuning

- [ ] Implement thresholding in `src/ncut/partition.py`.
- [ ] Implement NCut score calculation.
- [ ] Compare zero-crossing, median, and best-threshold methods.
- [ ] Tune `sigma_i`, `sigma_x`, and `radius` on test images.

### Prajwal: Graph and Solver

- [x] Set up repository structure.
- [x] Implement initial graph weight construction.
- [x] Implement initial Fiedler vector solver.
- [x] Improve graph validation and sparse matrix handling.
- [ ] Upgrade solver using SciPy sparse eigen-solvers.

### Saksham: Preprocessing and Robustness

- [x] Implement image loading in `src/ncut/utils.py`.
- [x] Add grayscale/RGB conversion.
- [x] Add image normalization.
- [x] Add resizing/downsampling for memory safety.
- [ ] Add error handling for disconnected or invalid graphs.

### Miraj: Visualization and Results

- [ ] Implement plotting in `src/ncut/visualize.py`.
- [ ] Create original-vs-mask-vs-overlay result views.
- [ ] Curate simple and complex test images.
- [ ] Prepare final result screenshots.
- [ ] Help maintain README and presentation material.
