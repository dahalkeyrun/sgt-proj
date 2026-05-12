Based on the project structure and the 3-week schedule we've established, here is a `TASK.md` file. You can commit this to the root of your repository so that **Kiran, Saksham, and Miraj** can easily track their progress alongside yours.

---

# 📋 Project Tasks: Sparse Spectral Image Segmentation

This document tracks the progress of our 6th-semester COMP 323 project. We are implementing the **Shi and Malik (2000)** Normalized Cuts algorithm as our first milestone.

## 🏗 Project Directory Structure

```text
spectral-segmentation-v1/
├── data/ (Input/Output images)
├── src/
│   ├── main.py        # Orchestration (Prajwal)
│   ├── preprocess.py  # Image handling (Saksham)
│   ├── engine.py      # Laplacian construction (Prajwal)
│   ├── solver.py      # Eigen-solver (Prajwal)
│   ├── logic.py       # Bipartitioning (Kiran)
│   └── utils.py       # Visualization (Miraj)
├── configs/           # Parameter tuning (Kiran)
└── requirements.txt   # Dependencies (Saksham)

```

---

## 👤 Kiran (Logic & Tuning)

* [ ] **Research:** Define $\sigma_I, \sigma_X$, and $r$ values from the Shi-Malik paper.
* [ ] **Bipartitioning:** Develop logic to threshold the Fiedler vector (Median/Zero crossing).
* [ ] **Ncut Calculation:** Implement formula to measure the quality of the split.
* [ ] **Tuning:** Experiment with `configs/params.yaml` to find optimal settings for various images.

## 👤 Prajwal (Backend Engine)

* [ ] **Environment:** Setup Git repository and standardized directory structure.
* [ ] **Weights:** Implement the Gaussian Kernel logic for pixel similarity ($w_{ij}$).
* [ ] **Laplacian:** Construct the Degree Matrix ($D$) and Laplacian ($L$) in `src/engine.py`.
* [ ] **Solver:** Integrate iterative Eigen-solver (Lanczos/ARPACK) in `src/solver.py`.
* [ ] **Integration:** Connect all modules in `src/main.py`.

## 👤 Saksham (Data & Optimization)

* [ ] **Preprocessing:** Write `src/preprocess.py` (Grayscale, Normalization).
* [ ] **Memory Safety:** Implement image resizing/downsampling logic.
* [ ] **Sparse Matrix:** Ensure $W$ and $L$ use CSR/CSC sparse formats to save RAM.
* [ ] **Robustness:** Add error handling for disconnected graph components.
* [ ] **Benchmarking:** Profile memory usage during matrix construction.

## 👤 Miraj (Data & Visualization)

* [ ] **Dataset:** Curate test images (simple shapes and complex backgrounds) in `data/input/`.
* [ ] **Mapping:** Create functions to map 1D eigenvectors back to the 2D image grid.
* [ ] **Visualization:** Develop `src/utils.py` for heatmaps and binary mask overlays.
* [ ] **Results:** Compile comparison charts (Original vs. Ncut result).
* [ ] **Docs:** Maintain the README and prepare the final presentation.

---

