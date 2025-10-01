#!/usr/bin/env python3
"""
Decision Tree (from scratch) + Learning Curve for 1.7

Usage:
    python dt_learning_curve.py --data Dbig.txt --seed 42

- Reads a plaintext dataset with lines: x1 x2 y  (whitespace-separated)
- Randomly permutes the 10,000 items
- Splits into train-candidate (8192) and test (rest)
- Builds nested training sets: 32, 128, 512, 2048, 8192
- Trains a decision tree (info gain ratio, thresholds at training values, tie-break predicts y=1)
- Reports table: n, #nodes, test error
- Saves: learning_curve.png and 5 decision boundary plots

Constraints from instructions:
- No scikit-learn; everything implemented from scratch.
- Split rule: feature j with threshold c using x_j >= c, left is "then" branch, right is "else".
- Thresholds c must be values present in the training set for that node.
- Skip candidate splits with zero split information (no separation).
- Choose split by information gain ratio; tie break arbitrarily.
- Stopping conditions:
    * Node is empty (shouldn't usually happen, but handle defensively)
    * All valid splits have zero gain ratio (or no valid splits)
    * Parent node entropy is zero (pure labels)
- If no majority class in a leaf, predict y=1 (tie-break).

Outputs:
- results_table.csv
- results_table.md
- learning_curve.png
- decision_boundary_n0032.png, ... n8192.png

Author: (you)
"""
import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os

import matplotlib.pyplot as plt
import numpy as np


# -------------------------- Utilities --------------------------

def entropy_of_labels(y: np.ndarray) -> float:
    """Binary entropy H(Y) with labels in {0,1}."""
    if y.size == 0:
        return 0.0
    p1 = np.mean(y == 1)
    p0 = 1.0 - p1
    if p0 <= 0 or p1 <= 0:
        return 0.0
    return - (p0 * math.log(p0, 2) + p1 * math.log(p1, 2))


def split_info(n_left: int, n_right: int) -> float:
    """Split information (intrinsic info) for a binary split."""
    n = n_left + n_right
    if n == 0 or n_left == 0 or n_right == 0:
        return 0.0
    pl = n_left / n
    pr = n_right / n
    return - (pl * math.log(pl, 2) + pr * math.log(pr, 2))


@dataclass
class Node:
    is_leaf: bool
    prediction: int
    feature: Optional[int] = None  # 0 or 1
    threshold: Optional[float] = None
    left: Optional["Node"] = None   # x_j >= c  (then-branch)
    right: Optional["Node"] = None  # else-branch
    n_samples: int = 0
    n_pos: int = 0


class DecisionTree:
    """Binary decision tree for 2D features using Information Gain Ratio."""

    def __init__(self):
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(X, y)

    def predict_row(self, x: np.ndarray) -> int:
        node = self.root
        while node and not node.is_leaf:
            j = node.feature
            c = node.threshold
            if x[j] >= c:
                node = node.left
            else:
                node = node.right
        # Fallback, though node should be leaf
        return node.prediction if node else 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_row(x) for x in X], dtype=int)

    def count_nodes(self) -> int:
        def _count(n: Optional[Node]) -> int:
            if n is None:
                return 0
            return 1 + _count(n.left) + _count(n.right)
        return _count(self.root)

    # ----------------- Internal: training a node -----------------

    def _build_tree(self, X: np.ndarray, y: np.ndarray) -> Node:
        node = Node(is_leaf=False, prediction=1, n_samples=len(y), n_pos=int(np.sum(y == 1)))

        # Leaf prediction: majority class; tie -> y=1
        n1 = node.n_pos
        n0 = node.n_samples - n1
        if n1 > n0:
            node.prediction = 1
        elif n0 > n1:
            node.prediction = 0
        else:
            node.prediction = 1  # tie-break

        # Stopping: empty or pure
        if node.n_samples == 0:
            node.is_leaf = True
            return node
        if entropy_of_labels(y) == 0.0:
            node.is_leaf = True
            return node

        # Enumerate candidate splits (feature j in {0,1}, threshold c among unique values in X[:,j])
        best = None  # (gain_ratio, j, c, partitions)
        H_parent = entropy_of_labels(y)

        for j in (0, 1):
            values = np.unique(X[:, j])
            for c in values:
                # Split rule: x_j >= c goes left; else right
                left_idx = np.where(X[:, j] >= c)[0]
                right_idx = np.where(X[:, j] < c)[0]
                nL, nR = len(left_idx), len(right_idx)

                # Skip if split info is zero (i.e., one side empty)
                SI = split_info(nL, nR)
                if SI == 0.0:
                    continue

                # Child entropies
                HL = entropy_of_labels(y[left_idx])
                HR = entropy_of_labels(y[right_idx])
                weighted_child_entropy = (nL / (nL + nR)) * HL + (nR / (nL + nR)) * HR

                IG = H_parent - weighted_child_entropy
                # If IG is zero, GR is zero; keep evaluating others
                GR = IG / SI if SI > 0 else 0.0

                if best is None or GR > best[0]:
                    best = (GR, j, float(c), left_idx, right_idx)

        # If no valid split or best GR <= 0, make leaf
        if best is None or best[0] <= 0.0:
            node.is_leaf = True
            return node

        # Otherwise, split and recurse
        _, j_best, c_best, left_idx, right_idx = best
        node.feature = j_best
        node.threshold = c_best

        node.left = self._build_tree(X[left_idx], y[left_idx])
        node.right = self._build_tree(X[right_idx], y[right_idx])
        return node


# -------------------------- Experiment --------------------------

def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            x1, x2, y = parts
            xs.append((float(x1), float(x2)))
            ys.append(int(float(y)))
    X = np.array(xs, dtype=float)
    y = np.array(ys, dtype=int)
    return X, y


def decision_boundary_plot(model: DecisionTree, X: np.ndarray, y: np.ndarray, title: str, outpath: str):
    # Define grid bounds
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 400),
        np.linspace(x2_min, x2_max, 400),
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = model.predict(grid).reshape(xx1.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx1, xx2, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5])
    # Plot points
    y0 = y == 0
    y1 = y == 1
    plt.scatter(X[y0, 0], X[y0, 1], s=10, label="y=0")
    plt.scatter(X[y1, 0], X[y1, 1], s=10, marker="x", label="y=1")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Dbig.txt", help="Path to Dbig.txt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation")
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    n = len(y)
    assert n >= 10000, "Dbig.txt should have 10,000 items per instructions."

    rng = random.Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)

    # Candidate train = first 8192 from permutation; test = rest
    train_candidate_idx = indices[:8192]
    test_idx = indices[8192:]

    X_train_cand, y_train_cand = X[train_candidate_idx], y[train_candidate_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Nested training sets using the SAME permutation order
    sizes = [32, 128, 512, 2048, 8192]
    results = []

    # Ensure output dir is current; images saved alongside script
    for n_train in sizes:
        X_train = X_train_cand[:n_train]
        y_train = y_train_cand[:n_train]

        tree = DecisionTree()
        tree.fit(X_train, y_train)

        # Test error
        y_pred = tree.predict(X_test)
        err = float(np.mean(y_pred != y_test))

        # Node count
        nodes = tree.count_nodes()

        results.append((n_train, nodes, err))

        # Decision boundary plot (using TRAIN set for visualization clarity)
        out_img = f"decision_boundary_n{n_train:04d}.png"
        decision_boundary_plot(tree, X_train, y_train,
                               title=f"Decision Boundary (n={n_train})",
                               outpath=out_img)

    # Save results table
    with open("results_table.csv", "w") as f:
        f.write("n,num_nodes,test_error\n")
        for n_train, nodes, err in results:
            f.write(f"{n_train},{nodes},{err:.6f}\n")

    # Also save a Markdown table for easy LaTeX/Markdown inclusion
    with open("results_table.md", "w") as f:
        f.write("| n | #nodes | test error |\n")
        f.write("|---:|------:|-----------:|\n")
        for n_train, nodes, err in results:
            f.write(f"| {n_train} | {nodes} | {err:.6f} |\n")

    # Learning curve plot n vs err
    ns = [r[0] for r in results]
    errs = [r[2] for r in results]

    plt.figure(figsize=(6, 5))
    plt.plot(ns, errs, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Training size n (log2 scale)")
    plt.ylabel("Test error")
    plt.title("Learning Curve (Decision Tree)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    plt.close()

    # Console printout
    print("Results (n, #nodes, test_error):")
    for row in results:
        print(row)
    print("\nSaved: results_table.csv, results_table.md, learning_curve.png, and 5 decision boundary images.")


if __name__ == "__main__":
    main()
