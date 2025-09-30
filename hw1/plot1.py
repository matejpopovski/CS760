#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import from your existing tree.py (the one I sent earlier)
from tree import read_txt, build_tree, predict_one

def plot_decision_boundary(data_path, out_png):
    # Load data
    X_list, y_list = read_txt(data_path)
    X = np.array(X_list, dtype=float)   # shape (n,2)
    y = np.array(y_list, dtype=int)     # shape (n,)

    # Train tree
    idxs = list(range(len(X)))
    tree = build_tree(X_list, y_list, idxs)

    # Grid for boundary
    pad = 0.05
    x_min, x_max = X[:,0].min() - pad, X[:,0].max() + pad
    y_min, y_max = X[:,1].min() - pad, X[:,1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    # Predict over grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([predict_one(tree, [p[0], p[1]]) for p in grid], dtype=int)
    Z = Z.reshape(xx.shape)

    # Plot decision regions + data
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, levels=1, alpha=0.35)  # decision regions
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor="k", s=20)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Decision Boundary — {data_path}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 plot1.py <datafile.txt>")
        sys.exit(1)
    data_path = sys.argv[1]
    base = data_path.rsplit(".", 1)[0]
    out_png = f"{base}_decision_boundary.png"
    plot_decision_boundary(data_path, out_png)
