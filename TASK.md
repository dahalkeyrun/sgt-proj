# Project Tasks: Sparse Spectral Image Segmentation

This document tracks progress for the 6th-semester Graph Theory mini-project:
**Image Segmentation using Normalized Cuts**, based on Shi and Malik (2000).

---

## Current Project Structure

```
sgt/
+-- README.md
+-- TASK.md
+-- requirements.txt
+-- app.py                             # Flask backend (DONE)
+-- backend_frontend_integration.md
+-- agent.md                           # Architecture audit spec
+-- data/input/
+-- scripts/
|   +-- run_segmentation.py            # CLI pipeline (DONE)
+-- src/
|   +-- ncut/
|       +-- __init__.py                # Empty — needs exports
|       +-- graph.py                   # DONE
|       +-- ncut.py                    # PARTIAL — power iteration only, no eigsh
|       +-- partition.py               # DONE
|       +-- recursive.py               # STUB only
|       +-- utils.py                   # DONE
|       +-- visualize.py               # DONE (console + PIL + restored)
+-- templates/
|   +-- index.html                     # Frontend UI (DONE, with canvases)
+-- tests/
    +-- test_graph.py                  # Diagnostic script, not pytest
    +-- test_image_pipeline.py         # Diagnostic script, not pytest
    +-- test_ncut.py                   # Diagnostic script, not pytest
```

---

## What's Done ✓

### All module bugs fixed
- [x] Imports corrected across all modules
- [x] Typos and dead code removed
- [x] `visualize.py` fully restored (console summary, ASCII heatmap/mask, PNG saves)
- [x] `run_segmentation.py` fully restored and functional

### Full pipeline live via Flask + frontend
- [x] `GET /` serves `index.html`
- [x] `POST /api/segment` accepts base64 image + params → returns RGBA arrays for canvases
- [x] Frontend renders heatmap, bisection mask, foreground overlay in real-time
- [x] Diagnostics display: nodes, edges, NCut score, elapsed time
- [x] Error handling on backend (400/500)

### src/ncut/graph.py — COMPLETE
- [x] Build graph from image pixels
- [x] Gaussian weights using intensity and spatial distance
- [x] Sparse CSR-like arrays + scipy.csr_matrix conversion
- [x] Degree vector
- [x] Parameter validation (image shape, sigma_i, sigma_x, radius)
- [x] Self-edges removed (W[i,i] = 0)
- [x] Strict radius-boundary (dist_x_sq < radius_sq, exclusive)
- [x] Grayscale and RGB support

### src/ncut/ncut.py — PARTIAL (core gap: eigenproblem)
- [x] D^-1 W multiplication
- [x] Fiedler vector via power iteration
- [x] Zero-degree handling
- [x] Solver input validation
- [x] Verified on synthetic left/right image
- [ ] **Replace manual iteration with `scipy.sparse.linalg.eigsh`**
- [ ] **Solve generalized eigenproblem: (D - W)y = λ D y**
- [ ] Return both eigenvalue and eigenvector
- [ ] Add convergence/error reporting

### src/ncut/partition.py — COMPLETE
- [x] Threshold by median
- [x] Threshold by mean
- [x] Binary partition mask
- [x] NCut score calculation
- [x] Input validation
- [ ] Threshold by zero-crossing (optional enhancement)
- [ ] Threshold search over Fiedler vector (optional enhancement)

### src/ncut/recursive.py — STUB ONLY (top priority gap)
- [x] `compute_partition_for_subgraph` wrapper
- [x] `build_label_map` helper
- [ ] `extract_subgraph(weight_matrix, node_indices)` — slice CSR into sub-adjacency matrix
- [ ] `recursive_split(...)` — actual recursion with stopping conditions
- [ ] `recursive_partition(...)` — public entry point
- [ ] Stopping conditions: max depth, min segment size, max NCut, min eigenvector stability
- [ ] Return final 2D label map
- [ ] Tests

### src/ncut/utils.py — COMPLETE
- [x] Load image from disk
- [x] Grayscale/RGB conversion
- [x] Normalize to [0, 1]
- [x] Resize/downsample
- [x] Reshape vectors to image grid
- [x] flatten_image helper
- [ ] Guard against disconnected/invalid graphs (solid-color → zero degree → garbage Fiedler)

### src/ncut/visualize.py — COMPLETE (console + PIL output)
- [x] Console summary printing
- [x] ASCII Fiedler heatmap
- [x] ASCII partition mask
- [x] Save mask as PNG
- [x] Save Fiedler heatmap as PNG
- [x] Save comparison (original + mask)
- [ ] Matplotlib plots: original image, Fiedler heatmap, binary mask, overlay (2×2 figure)

### scripts/run_segmentation.py — COMPLETE
- [x] CLI with argparse (image_path, sigma_i, sigma_x, radius, threshold, output)
- [x] Full pipeline: load → build_graph → find_fiedler → partition → visualize
- [x] Save outputs (mask, heatmap, comparison)
- [x] Print summary info

### app.py (Flask backend) — COMPLETE
- [x] `GET /` serves index.html
- [x] `POST /api/segment` accepts base64 image + params
- [x] Returns JSON: width, height, nodes, edges, ncut_score, threshold, elapsed_ms
- [x] Returns RGBA flat arrays for: heatmap, bisection mask, original nodes, foreground
- [x] Fiedler → blue-white-red colormap
- [x] Mask → black/white RGBA
- [x] Foreground → masked transparency
- [x] Error handling

### templates/index.html — COMPLETE
- [x] UI with image upload, sliders, process button
- [x] Canvas elements for heatmap, bisection, foreground
- [x] Diagnostics display section

### src/ncut/__init__.py — EMPTY
- [ ] Export main public API functions once stable

### Tests — DIAGNOSTIC SCRIPTS, NOT PYTEST
- [ ] Convert `test_graph.py`, `test_ncut.py`, `test_image_pipeline.py` to proper pytest
  - Rename `run_*` → `test_*` so pytest collects them
- [ ] Remove `sys.path.append(...)` from all test files
- [ ] Add `test_partition.py`
- [ ] Add integration test (full pipeline, tiny mock image)
- [ ] Add tests for `recursive.py`

---

## Next Steps (ordered by priority)

### 1. Complete `recursive.py`
Missing functions (see architecture audit in `agent.md`):
- `extract_subgraph(weight_matrix, node_indices)` — slice CSR into sub-adjacency matrix
- `recursive_split(...)` — actual recursion with stopping conditions (max depth, min segment size, max NCut, stability)
- `recursive_partition(...)` — public entry point

### 2. Convert tests to proper pytest
- Rename `run_*` → `test_*` in all 3 test files
- Remove `sys.path.append` usage
- Add `test_partition.py`
- Add integration test

### 3. Upgrade eigensolver in `ncut.py`
```python
from scipy.sparse.linalg import eigsh
# solve (D - W) y = λ D y  →  eigsh on L_sym
```
- This is the only remaining item in Prajwal's task split.

### 4. Export public API from `__init__.py`
Once module APIs stabilize, export key functions so callers don't need to know internal module names.

### 5. Add `/api/segment` support for recursive multi-region (after #1)
Accept a `mode` parameter (`"bipartition"` vs `"recursive"`) and return a label map for multi-region coloring instead of a binary mask.

### 6. Error handling for disconnected graphs in `utils.py`
Guard against images that produce a fully disconnected graph (e.g., solid-color). Check `degree_vector` before calling `find_fiedler_vector`.

---

## Team Task Split

| Who | What |
|-----|------|
| **Kiran** | Convert tests to pytest (#2) |
| **Prajwal** | `recursive.py` functions (#1) + `eigsh` upgrade (#3) |
| **Saksham** | Disconnected graph guard in `utils.py` (#6) |
| **Miraj** | `__init__.py` exports (#4) + test result screenshots |
