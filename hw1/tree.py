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

CLI:
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

def _entropy(labels: List[int]) -> float:
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

def _split_info(n_left: int, n_right: int, n_total: int) -> float:
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

# -------------------- Tree node --------------------

class _Node:
    def __init__(
        self,
        is_leaf: bool,
        pred: Optional[int] = None,
        feat: Optional[int] = None,
        thr: Optional[float] = None,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
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

# -------------------- DecisionTree class --------------------

class DecisionTree:
    """
    Minimal API:
      - fit(X, y)
      - predict(X) -> list[int]
      - score(X, y) -> float accuracy
      - export_text() -> string (pretty tree)
      - num_nodes_  (total nodes)
      - num_leaves_
      - num_internals_
      - depth_
    """
    def __init__(self):
        self.root_: Optional[_Node] = None
        self.num_leaves_ = 0
        self.num_internals_ = 0
        self.num_nodes_ = 0
        self.depth_ = 0

    # ---------- helpers ----------
    @staticmethod
    def _majority_label(labels: List[int]) -> int:
        ones = sum(labels)
        zeros = len(labels) - ones
        if ones > zeros:
            return 1
        if zeros > ones:
            return 0
        return 1  # tie -> predict 1

    @staticmethod
    def _split_indices(X: List[List[float]], idxs: List[int], j: int, c: float) -> Tuple[List[int], List[int]]:
        """Return (left, right) where left = indices with x_j >= c (Yes), right = otherwise (No)."""
        left, right = [], []
        for i in idxs:
            if X[i][j] >= c:
                left.append(i)
            else:
                right.append(i)
        return left, right

    def _build(self, X: List[List[float]], y: List[int], idxs: List[int]) -> _Node:
        """Recursively build the decision tree according to assignment rules."""
        # Empty node -> leaf predicting 1 (tie/empty rule)
        if not idxs:
            return _Node(is_leaf=True, pred=1, n=0, n_pos=0, H=0.0)

        labels = [y[i] for i in idxs]
        n_total = len(labels)
        pos = sum(labels)
        H_parent = _entropy(labels)

        # Pure -> leaf
        if H_parent == 0.0:
            return _Node(is_leaf=True, pred=labels[0], n=n_total, n_pos=pos, H=H_parent)

        # Enumerate candidate splits
        best = None  # tuple: (key, GR, IG, j, c, left, right)
        for j in (0, 1):
            values = sorted({X[i][j] for i in idxs})
            for c in values:
                left, right = self._split_indices(X, idxs, j, c)
                nL, nR = len(left), len(right)
                if nL == 0 or nR == 0:
                    continue  # degenerate

                SI = _split_info(nL, nR, n_total)
                if SI == 0.0:
                    continue  # skip per spec

                H_left = _entropy([y[i] for i in left])
                H_right = _entropy([y[i] for i in right])
                H_split = (nL / n_total) * H_left + (nR / n_total) * H_right

                IG = H_parent - H_split
                if IG <= 0.0:
                    continue

                GR = IG / SI

                # Prefer larger GR; tie-break by larger IG, then smaller feature index, then smaller threshold.
                key = (GR, IG, -j, -c)
                if best is None or key > best[0]:
                    best = (key, GR, IG, j, c, left, right)

        # No valid improving split -> make leaf
        if best is None:
            return _Node(is_leaf=True, pred=self._majority_label(labels), n=n_total, n_pos=pos, H=H_parent)

        # Unpack best split and recurse
        _, GR, IG, j, c, left, right = best
        left_node = self._build(X, y, left)
        right_node = self._build(X, y, right)
        return _Node(is_leaf=False, pred=None, feat=j, thr=c, left=left_node, right=right_node, n=n_total, n_pos=pos, H=H_parent)

    # ---------- public API ----------
    def fit(self, X: List[List[float]], y: List[int]) -> "DecisionTree":
        idxs = list(range(len(X)))
        self.root_ = self._build(X, y, idxs)
        # stats
        leaves, internals = self._count_nodes(self.root_)
        self.num_leaves_ = leaves
        self.num_internals_ = internals
        self.num_nodes_ = leaves + internals
        self.depth_ = self._max_depth(self.root_)
        return self

    def _predict_one(self, x: List[float]) -> int:
        node = self.root_
        assert node is not None
        cur = node
        while not cur.is_leaf:
            if x[cur.feat] >= cur.thr:
                cur = cur.left
            else:
                cur = cur.right
        return cur.pred

    def predict(self, X: List[List[float]]) -> List[int]:
        return [self._predict_one(x) for x in X]

    def score(self, X: List[List[float]], y: List[int]) -> float:
        yh = self.predict(X)
        return sum(int(a == b) for a, b in zip(yh, y)) / len(y)

    # ---------- stats & printing ----------
    def _count_nodes(self, node: _Node) -> Tuple[int, int]:
        """Return (num_leaves, num_internal)."""
        if node.is_leaf:
            return (1, 0)
        lL, lI = self._count_nodes(node.left)
        rL, rI = self._count_nodes(node.right)
        return (lL + rL, lI + rI + 1)

    def _max_depth(self, node: _Node) -> int:
        if node.is_leaf:
            return 1
        return 1 + max(self._max_depth(node.left), self._max_depth(node.right))

    def _export_lines(self, node: _Node, prefix: str = "Root: ") -> List[str]:
        if node.is_leaf:
            if node.n is not None and node.n_pos is not None:
                return [f"{prefix}y = {node.pred}  [n={node.n}, pos={node.n_pos}]"]
            return [f"{prefix}y = {node.pred}"]
        feat_name = f"x{node.feat + 1}"
        lines = [f"{prefix}{feat_name} >= {node.thr}?"]
        lines += self._export_lines(node.left, prefix="  Yes: ")
        lines += self._export_lines(node.right, prefix="  No:  ")
        return lines

    def export_text(self) -> str:
        assert self.root_ is not None
        return "\n".join(self._export_lines(self.root_))

# -------------------- CLI --------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python tree.py <datafile.txt>")
        sys.exit(1)

    data_path = sys.argv[1]
    X, y = read_txt(data_path)

    clf = DecisionTree().fit(X, y)
    print(clf.export_text())

    acc = clf.score(X, y)
    print(
        f"\n[STATS] accuracy={acc:.4f}, "
        f"leaves={clf.num_leaves_}, internals={clf.num_internals_}, "
        f"depth={clf.depth_}, n={len(X)}"
    )

if __name__ == "__main__":
    main()
