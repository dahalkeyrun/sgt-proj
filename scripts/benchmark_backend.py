#!/usr/bin/env python3
"""Simple backend benchmark for the Normalized Cuts pipeline."""

import argparse
import sys
import os
import time
from statistics import mean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ncut.backend import Backend
from ncut.graph import build_weight_matrix
from ncut.ncut import find_fiedler_vector
from ncut.partition import partition_graph
from ncut.utils import load_image


def synchronize_backend(backend):
    if backend != Backend.GPU:
        return

    import cupy as cp

    cp.cuda.Stream.null.synchronize()


def run_once(image, args, backend):
    start = time.perf_counter()
    weight_matrix, degree_vector = build_weight_matrix(
        image,
        sigma_i=args.sigma_i,
        sigma_x=args.sigma_x,
        radius=args.radius,
        backend=backend,
    )
    synchronize_backend(backend)
    graph_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    fiedler_vector = find_fiedler_vector(
        weight_matrix.data,
        weight_matrix.indices,
        weight_matrix.indptr,
        degree_vector,
        backend=backend,
    )
    eigensolver_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    partition_result = partition_graph(weight_matrix, fiedler_vector, backend=backend)
    partition_ms = (time.perf_counter() - start) * 1000.0

    total_ms = graph_ms + eigensolver_ms + partition_ms
    return {
        "nodes": weight_matrix.shape[0],
        "edges": weight_matrix.nnz,
        "graph_ms": graph_ms,
        "eigensolver_ms": eigensolver_ms,
        "partition_ms": partition_ms,
        "total_ms": total_ms,
        "threshold": partition_result["threshold"],
        "ncut": partition_result["ncut"],
    }


def summarize_runs(results):
    keys = ["graph_ms", "eigensolver_ms", "partition_ms", "total_ms"]
    return {
        key: {
            "mean": mean(result[key] for result in results),
            "best": min(result[key] for result in results),
        }
        for key in keys
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU/GPU segmentation backend")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--sigma-i", type=float, default=0.1)
    parser.add_argument("--sigma-x", type=float, default=4.0)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=30)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup runs before measurement")
    parser.add_argument("--runs", type=int, default=1, help="Measured runs")
    args = parser.parse_args()

    backend = Backend.from_string(args.backend)
    warmup_runs = args.warmup if args.warmup is not None else (1 if backend == Backend.GPU else 0)
    image = load_image(args.image_path, mode="grayscale", max_size=args.resolution)

    for index in range(warmup_runs):
        print(f"warmup_run={index + 1}")
        run_once(image, args, backend)

    results = []
    for index in range(args.runs):
        result = run_once(image, args, backend)
        results.append(result)
        print(f"run={index + 1}")
        print(f"backend={backend.value}")
        print(f"nodes={result['nodes']}")
        print(f"edges={result['edges']}")
        print(f"graph_ms={result['graph_ms']:.2f}")
        print(f"eigensolver_ms={result['eigensolver_ms']:.2f}")
        print(f"partition_ms={result['partition_ms']:.2f}")
        print(f"total_ms={result['total_ms']:.2f}")
        print(f"threshold={result['threshold']:.6f}")
        print(f"ncut={result['ncut']:.6f}")

    if len(results) > 1:
        summary = summarize_runs(results)
        print("summary=measured_runs")
        for stage, values in summary.items():
            print(f"{stage}_mean={values['mean']:.2f}")
            print(f"{stage}_best={values['best']:.2f}")


if __name__ == "__main__":
    main()
