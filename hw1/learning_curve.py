import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

# >>> import your tree <<<
from tree import DecisionTree  # must provide fit(X,y), predict(X), and num_nodes_

# ---------------------- config ----------------------
RNG_SEED = 760
TRAIN_TARGET_SIZE = 8192
subset_sizes = [32, 128, 512, 2048, 8192]
outdir = Path("outputs_1_7")
outdir.mkdir(exist_ok=True, parents=True)
# ----------------------------------------------------

def load_txt(fname):
    arr = np.loadtxt(fname)
    X = arr[:, :2]
    y = arr[:, 2].astype(int)
    return X, y

def permute_and_split(X, y, train_size, seed=RNG_SEED):
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    Xp, yp = X[perm], y[perm]
    Xtr, ytr = Xp[:train_size], yp[:train_size]
    Xte, yte = Xp[train_size:], yp[train_size:]
    return Xtr, ytr, Xte, yte

def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)

def plot_decision_boundary(clf, X, y, title, outpath, padding=0.02, grid_step=400):
    # bounds
    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    dx = (x_max - x_min) * 0.05 + padding
    dy = (y_max - y_min) * 0.05 + padding
    x_min, x_max = x_min - dx, x_max + dx
    y_min, y_max = y_min - dy, y_max + dy

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_step),
        np.linspace(y_min, y_max, grid_step),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array(clf.predict(grid)).reshape(xx.shape)

    # plot
    plt.figure(figsize=(7,5.2), dpi=140)
    plt.contourf(xx, yy, Z, alpha=0.25, levels=2)
    plt.scatter(X[:,0], X[:,1], c=y, s=12, edgecolor='k', linewidth=0.2, cmap='viridis')
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

@dataclass
class Row:
    n: int
    nodes: int
    err: float

def main():
    X, y = load_txt("Dbig.txt")
    # IMPORTANT: Dbig.txt is sorted in the prompt; we randomize once before split.
    Xtr_full, ytr_full, Xte, yte = permute_and_split(X, y, TRAIN_TARGET_SIZE, seed=RNG_SEED)

    rows = []
    for n in subset_sizes:
        Xtr = Xtr_full[:n]
        ytr = ytr_full[:n]

        clf = DecisionTree()
        clf.fit(Xtr, ytr)

        yhat = clf.predict(Xte)
        err = error_rate(yte, yhat)
        # grab number of nodes (adapt if your tree uses a different name / method)
        num_nodes = getattr(clf, "num_nodes_", None)
        if num_nodes is None:
            # as a fallback, you can add a method clf.count_nodes()
            num_nodes = clf.count_nodes() if hasattr(clf, "count_nodes") else -1

        rows.append(Row(n=n, nodes=num_nodes, err=err))

        # decision boundary for this subset (plot uses the training points)
        plot_decision_boundary(
            clf, Xtr, ytr,
            title=f"Decision Boundary — n={n}",
            outpath=outdir / f"db_n={n}.png",
        )
        print(f"n={n:5d} | nodes={num_nodes:5d} | err={err:.4f}")

    # save table (csv)
    import csv
    with open(outdir / "learning_curve_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "num_nodes", "err_n"])
        for r in rows:
            w.writerow([r.n, r.nodes, f"{r.err:.4f}"])

    # plot learning curve
    plt.figure(figsize=(6.5,4.6), dpi=140)
    plt.plot([r.n for r in rows], [r.err for r in rows], marker="o")
    plt.xscale("log", base=2)  # nice spacing for 32→8192
    plt.xlabel("training size n (log2 scale)")
    plt.ylabel("test error  err_n")
    plt.title("Learning curve (Decision Tree on Dbig)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "learning_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
