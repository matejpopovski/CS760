#!/usr/bin/env python3
"""
Decision tree learner for 2D continuous features (x in R^2) and binary labels y in {0,1},
implemented from scratch to match the assignment rules:

- Candidate splits are of the form x_j >= c, with c chosen ONLY from training values
  present at the current node (no midpoints).
- Skip candidate splits with zero split information.
- Choose the split with maximum gain ratio (information gain / split info).
  Tie-breaks deterministically by (higher IG, smaller feature index, then smaller threshold).
- Stopping criteria for making a node into a leaf:
    * node is empty
    * node is pure (entropy(node) == 0)
    * no candidate split yields positive gain ratio
- Leaf prediction = majority class in that node (ties predict 1).
- The "Yes" branch is the THEN branch (condition true), "No" is the ELSE branch.

Usage:
    python tree.py <datafile.txt>

Data format (plaintext), one item per line:
    x1  x2  y
with whitespace separation.
"""

import math
import sys
from typing import List, Tuple, Optional

# -------------------- I/O --------------------

def read_txt(path: str) -> Tuple[List[List[float]], List[int]]:
    X, y = [], []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 3:
                raise ValueError(f"Bad line (expect 3 fields): {ln}")
            a, b, lab = parts
            X.append([float(a), float(b)])
            yi = int(lab)
            if yi not in (0, 1):
                raise ValueError(f"Label must be 0/1, got {yi} on line: {ln}")
            y.append(yi)
    return X, y

# -------------------- Math utils --------------------

def entropy(labels: List[int]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    p1 = sum(labels) / n
    p0 = 1.0 - p1
    e = 0.0
    if p1 > 0:
        e -= p1 * math.log(p1, 2)
    if p0 > 0:
        e -= p0 * math.log(p0, 2)
    return e

def split_info(n_left: int, n_right: int, n_total: int) -> float:
    """Split information (intrinsic value) for gain ratio."""
    if n_total == 0:
        return 0.0
    si = 0.0
    for k in (n_left, n_right):
        if k == 0:
            continue
        p = k / n_total
        si -= p * math.log(p, 2)
    return si

# -------------------- Tree structures --------------------

class Node:
    def __init__(
        self,
        is_leaf: bool,
        pred: Optional[int] = None,
        feat: Optional[int] = None,
        thr: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        n: Optional[int] = None,
        n_pos: Optional[int] = None,
        H: Optional[float] = None,
    ):
        self.is_leaf = is_leaf
        self.pred = pred
        self.feat = feat
        self.thr = thr
        self.left = left
        self.right = right
        self.n = n           # number of samples at node
        self.n_pos = n_pos   # number of positives at node
        self.H = H           # entropy at node

# -------------------- Core algorithm --------------------

def majority_label(labels: List[int]) -> int:
    ones = sum(labels)
    zeros = len(labels) - ones
    if ones > zeros:
        return 1
    if zeros > ones:
        return 0
    return 1  # tie -> predict 1

def split_indices(X: List[List[float]], idxs: List[int], j: int, c: float) -> Tuple[List[int], List[int]]:
    """Return (left, right) where left = indices with x_j >= c (Yes), right = otherwise (No)."""
    left, right = [], []
    for i in idxs:
        if X[i][j] >= c:
            left.append(i)
        else:
            right.append(i)
    return left, right

def build_tree(
    X: List[List[float]], y: List[int], idxs: List[int]
) -> Node:
    """Recursively build the decision tree according to assignment rules."""
    # Empty node -> leaf predicting 1 (tie/empty rule)
    if not idxs:
        return Node(is_leaf=True, pred=1, n=0, n_pos=0, H=0.0)

    labels = [y[i] for i in idxs]
    n_total = len(labels)
    pos = sum(labels)
    H_parent = entropy(labels)

    # Pure -> leaf
    if H_parent == 0.0:
        return Node(is_leaf=True, pred=labels[0], n=n_total, n_pos=pos, H=H_parent)

    # Enumerate candidate splits
    best = None  # tuple: (GR, IG, j, c, left, right)
    for j in (0, 1):
        values = sorted({X[i][j] for i in idxs})
        for c in values:
            left, right = split_indices(X, idxs, j, c)
            nL, nR = len(left), len(right)
            # defensively skip degenerate splits (all to one side)
            if nL == 0 or nR == 0:
                continue

            # Split info
            SI = split_info(nL, nR, n_total)
            if SI == 0.0:
                continue  # skip per spec

            # Entropies
            H_left = entropy([y[i] for i in left])
            H_right = entropy([y[i] for i in right])
            H_split = (nL / n_total) * H_left + (nR / n_total) * H_right

            IG = H_parent - H_split
            if IG <= 0.0:
                # No information gain -> GR won't be positive; skip
                continue

            GR = IG / SI

            # Keep the best by GR, tie-break by IG (higher), then feature index (smaller), then threshold (smaller)
            key = (GR, IG, -j, -c)  # invert j/c sign in key to prefer smaller j, c
            if best is None or key > best[0]:
                best = (key, GR, IG, j, c, left, right)

    # No valid improving split -> make leaf
    if best is None:
        return Node(is_leaf=True, pred=majority_label(labels), n=n_total, n_pos=pos, H=H_parent)

    # Unpack best split and recurse
    _, GR, IG, j, c, left, right = best
    left_node = build_tree(X, y, left)
    right_node = build_tree(X, y, right)
    return Node(is_leaf=False, pred=None, feat=j, thr=c, left=left_node, right=right_node, n=n_total, n_pos=pos, H=H_parent)

# -------------------- Utilities: predict, stats, pretty print --------------------

def predict_one(node: Node, x: List[float]) -> int:
    cur = node
    while not cur.is_leaf:
        if x[cur.feat] >= cur.thr:
            cur = cur.left
        else:
            cur = cur.right
    return cur.pred

def predict_all(node: Node, X: List[List[float]]) -> List[int]:
    return [predict_one(node, xi) for xi in X]

def accuracy(node: Node, X: List[List[float]], y: List[int]) -> float:
    yh = predict_all(node, X)
    return sum(int(a == b) for a, b in zip(yh, y)) / len(y)

def count_nodes(node: Node) -> Tuple[int, int]:
    """Return (num_leaves, num_internal)."""
    if node.is_leaf:
        return (1, 0)
    lL, lI = count_nodes(node.left)
    rL, rI = count_nodes(node.right)
    return (lL + rL, lI + rI + 1)

def max_depth(node: Node) -> int:
    if node.is_leaf:
        return 1
    return 1 + max(max_depth(node.left), max_depth(node.right))

def print_tree(node: Node, prefix: str = "Root: "):
    if node.is_leaf:
        # Show purity info to sanity-check leaves
        if node.n is not None and node.n_pos is not None:
            print(f"{prefix}y = {node.pred}  [n={node.n}, pos={node.n_pos}]")
        else:
            print(f"{prefix}y = {node.pred}")
        return
    feat_name = f"x{node.feat + 1}"
    # Keep full precision threshold (matches your earlier output style)
    print(f"{prefix}{feat_name} >= {node.thr}?")
    print_tree(node.left, prefix="  Yes: ")
    print_tree(node.right, prefix="  No:  ")

# -------------------- Main --------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python tree.py <datafile.txt>")
        sys.exit(1)

    data_path = sys.argv[1]
    X, y = read_txt(data_path)
    idxs = list(range(len(X)))

    tree = build_tree(X, y, idxs)
    print_tree(tree)

    # Useful stats for sanity
    acc = accuracy(tree, X, y)
    leaves, internals = count_nodes(tree)
    depth = max_depth(tree)
    print(f"\n[STATS] accuracy={acc:.4f}, leaves={leaves}, internals={internals}, depth={depth}, n={len(X)}")

if __name__ == "__main__":
    main()
