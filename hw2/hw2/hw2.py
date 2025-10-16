import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# -----------------------------
# 0) Hyperparameters
# -----------------------------
MAX_VOCAB = 10_000     # top 10k tokens by frequency (train-only)
MAX_ITER  = 1000

# -----------------------------
# 1) Load TSV data:  label<TAB>review
# -----------------------------
train = pd.read_csv("imdb_train.csv", sep="\t", names=["label", "text"], encoding="utf-8")
test  = pd.read_csv("imdb_test.csv",  sep="\t", names=["label", "text"],  encoding="utf-8")

# Labels: pos->1, neg->0
y_train = (train["label"] == "pos").astype(int).to_numpy()
y_test  = (test["label"]  == "pos").astype(int).to_numpy()

# -----------------------------
# 2) Tokenization per spec:
#    lowercase + replace ALL punctuation with spaces + split on spaces
# -----------------------------
punct_to_space = str.maketrans({c: " " for c in string.punctuation})
def preproc(s: str) -> str:
    # lowercase then replace punctuation with spaces
    return (s or "").lower().translate(punct_to_space)

# -----------------------------
# 3) Vectorize: build vocab on TRAIN ONLY, binary presence features
# -----------------------------
vect = CountVectorizer(
    preprocessor=preproc,
    tokenizer=str.split,      # strictly split on whitespace
    token_pattern=None,       # must be None when tokenizer is provided
    max_features=MAX_VOCAB,
    binary=True               # presence/absence (0/1)
)

X_train = vect.fit_transform(train["text"])
X_test  = vect.transform(test["text"])   # IMPORTANT: transform with SAME vocab

print(f"X_train shape: {X_train.shape} (expected ~25k x 10k)")
print(f"X_test  shape: {X_test.shape}")

# -----------------------------
# 4) Section 2.1 — Train (unregularized) logistic regression and evaluate
# -----------------------------
print("\n=== 2.1: Unregularized Logistic Regression ===")
try:
    clf_unreg = LogisticRegression(penalty="none", solver="lbfgs", max_iter=MAX_ITER)
    clf_unreg.fit(X_train, y_train)
except Exception:
    # Fallback: approximate "no reg" with huge C and liblinear
    clf_unreg = LogisticRegression(penalty="l2", C=1e9, solver="liblinear", max_iter=MAX_ITER)
    clf_unreg.fit(X_train, y_train)

y_pred_unreg = clf_unreg.predict(X_test)
acc_unreg = accuracy_score(y_test, y_pred_unreg)
print(f"Test accuracy (unregularized): {acc_unreg:.4f}")

# -----------------------------
# 5) Section 2.3 — L2 regularization with 4-fold CV over C ∈ [1e-4, 1e4]
# -----------------------------
print("\n=== 2.3: L2-regularized Logistic Regression with 4-fold CV ===")
Cs = np.logspace(-4, 4, 10)
cv_splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

clf_cv = LogisticRegressionCV(
    Cs=Cs,
    cv=cv_splitter,
    penalty="l2",
    solver="lbfgs",
    scoring="accuracy",
    max_iter=MAX_ITER,
    n_jobs=-1,
    refit=True
)
clf_cv.fit(X_train, y_train)

# Extract best C and best CV accuracy
best_C = float(clf_cv.C_[0])
# Pick whichever class key exists (usually 1 for binary)
cls_key = 1 if 1 in clf_cv.scores_ else next(iter(clf_cv.scores_))
cv_means = clf_cv.scores_[cls_key].mean(axis=0)   # mean over folds for each C
best_cv_acc = float(cv_means[np.argmax(cv_means)])

y_pred_cv = clf_cv.predict(X_test)
acc_cv = accuracy_score(y_test, y_pred_cv)

print(f"Best C from CV: {best_C:.4g}")
print(f"Best 4-fold CV accuracy (train folds): {best_cv_acc:.4f}")
print(f"Test accuracy (L2, best C): {acc_cv:.4f}")

print("\n=== 2.4 ===")
targets = ["the", "good", "bad"]

def weight(clf, vec, token):
    j = vec.vocabulary_.get(token)
    return None if j is None else float(clf.coef_[0, j])

for t in targets:
    w_u  = weight(clf_unreg, vect, t)
    w_l2 = weight(clf_cv,    vect, t)
    print(f"{t:>5}: unreg={w_u:+.6f}   L2={w_l2:+.6f}")

# === 2.5: Latent Semantic Analysis (LSA) ===
# Goal: tokens as samples; compute R^(LSA) = U_k Σ_k (shape: 10_000 × 10)
from sklearn.decomposition import TruncatedSVD

print("\n=== 2.5 ===")
# 1) Load unlabeled reviews (50k lines)
with open("imdb_unsup.txt", "r", encoding="utf-8") as f:
    unsup_lines = [line.rstrip("\n") for line in f]

# 2) Vectorize with the SAME training vocabulary (do NOT .fit again)
X_unsup = vect.transform(unsup_lines)            # shape ~ (50_000, 10_000)
print("X_unsup shape:", X_unsup.shape)
assert X_unsup.shape[1] == X_train.shape[1], "Vocab size mismatch"

# 3) Tokens as samples => run SVD on X_unsup.T (10_000 × 50_000, sparse)
svd = TruncatedSVD(n_components=10, algorithm="arpack")  # deterministic
R_lsa = svd.fit_transform(X_unsup.T)                     # U_k Σ_k
print("R_lsa shape (tokens × 10):", R_lsa.shape)

# 4) (Optional) peek at a few token embeddings to verify
for tok in ["the", "good", "bad"]:
    j = vect.vocabulary_.get(tok)
    if j is not None:
        print(f"{tok}: {R_lsa[j, :5]}")  # first 5 dims


# === 2.6: Continuous BoW (CBOW) using LSA token embeddings ===
# Pre-req: R_lsa (10000 x 10) from 2.5, X_train/X_test from 2.1

from sklearn.linear_model import LogisticRegressionCV

print("\n=== 2.6 ===")
# 1) Build document embeddings by summing token vectors
X_train_lsa = X_train.dot(R_lsa)   # shape: (25_000, 10)
X_test_lsa  = X_test.dot(R_lsa)    # shape: (25_000, 10)
print("X_train_lsa shape:", X_train_lsa.shape)
print("X_test_lsa  shape:", X_test_lsa.shape)

# 2) 4-fold CV logistic regression (L2), C ∈ [1e-4, 1e4] (10 log-spaced)
Cs = np.logspace(-4, 4, 10)
cv_splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

clf_cbow = LogisticRegressionCV(
    Cs=Cs,
    cv=cv_splitter,
    penalty="l2",
    solver="lbfgs",
    scoring="accuracy",
    max_iter=1000,
    n_jobs=-1,
    refit=True
)
clf_cbow.fit(X_train_lsa, y_train)

best_C_cbow = float(clf_cbow.C_[0])
# scores_ key may be 0 or 1; pick whatever exists
cls_key = 1 if 1 in clf_cbow.scores_ else next(iter(clf_cbow.scores_))
cv_means_cbow = clf_cbow.scores_[cls_key].mean(axis=0)
best_cv_acc_cbow = float(cv_means_cbow[np.argmax(cv_means_cbow)])

y_pred_cbow = clf_cbow.predict(X_test_lsa)
test_acc_cbow = accuracy_score(y_test, y_pred_cbow)

print(f"=== 2.6 (CBOW) ===")
print(f"Best C: {best_C_cbow:.4g}")
print(f"Best 4-fold CV accuracy: {best_cv_acc_cbow:.4f}")
print(f"Test accuracy: {test_acc_cbow:.4f}")

# === 2.9: GloVe-based CBoW (300-D) ===========================================
# Input file format (glove.csv): token<TAB>v1 v2 ... v300  (space-delimited floats)

GLOVE_PATH = "glove.csv"
GLOVE_DIM = 300

def load_glove_tab(path: str, dim: int = 300) -> dict[str, np.ndarray]:
    glove = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                tok, vec_str = line.split("\t", 1)
            except ValueError:
                continue  # skip malformed lines
            vec = np.fromstring(vec_str, sep=" ", dtype=np.float32)
            if vec.size == dim:
                glove[tok] = vec
    return glove

print("Loading GloVe...")
glove = load_glove_tab(GLOVE_PATH, dim=GLOVE_DIM)
print(f"GloVe entries loaded: {len(glove):,}")

# Build embedding matrix aligned to our 10k-word vocabulary
V = len(vect.vocabulary_)
R_glove = np.zeros((V, GLOVE_DIM), dtype=np.float32)  # OOV -> zero vector
covered = 0
for tok, j in vect.vocabulary_.items():
    vec = glove.get(tok)
    if vec is not None:
        R_glove[j] = vec
        covered += 1
print(f"Vocab covered by GloVe: {covered}/{V} ({covered / V:.1%})")

# Construct 300-D document embeddings by summing token vectors
X_train_glove = X_train.dot(R_glove)   # (25_000, 300)
X_test_glove  = X_test.dot(R_glove)    # (25_000, 300)
print("X_train_glove shape:", X_train_glove.shape)
print("X_test_glove  shape:", X_test_glove.shape)

# 4-fold CV logistic regression (L2), C in [1e-4, 1e4]
Cs = np.logspace(-4, 4, 10)
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

clf_glove = LogisticRegressionCV(
    Cs=Cs,
    cv=cv,
    penalty="l2",
    solver="lbfgs",
    scoring="accuracy",
    max_iter=1000,
    n_jobs=-1,
    refit=True,
)

clf_glove.fit(X_train_glove, y_train)

best_C_glove = float(clf_glove.C_[0])
key = 1 if 1 in clf_glove.scores_ else next(iter(clf_glove.scores_))
cv_means = clf_glove.scores_[key].mean(axis=0)
best_cv_acc_glove = float(cv_means[np.argmax(cv_means)])
test_acc_glove = accuracy_score(y_test, clf_glove.predict(X_test_glove))

print("=== 2.9 (GloVe CBOW) ===")
print(f"Best C: {best_C_glove:.4g}")
print(f"Best 4-fold CV accuracy: {best_cv_acc_glove:.4f}")
print(f"Test accuracy: {test_acc_glove:.4f}")


# === 2.10: Learning curves (BoW, LSA-CBOW, GloVe-CBOW) ======================
import matplotlib.pyplot as plt

# sizes requested (balanced: half pos, half neg)
sizes = [8, 40, 200, 1000, 5000, 25000]
Cs = np.logspace(-4, 4, 10)
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

def balanced_indices(y, n, seed=0):
    assert n % 2 == 0, "n must be even for balance"
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; rng.shuffle(pos)
    neg = np.where(y == 0)[0]; rng.shuffle(neg)
    n_per = n // 2
    idx = np.concatenate([pos[:n_per], neg[:n_per]])
    rng.shuffle(idx)
    return idx

def cv_lr_test_acc(X_sub, y_sub, X_test_rep, y_test, n_train):
    # small n: liblinear is stabler; larger n: lbfgs is faster
    solver = "liblinear" if n_train <= 200 else "lbfgs"
    clf = LogisticRegressionCV(
        Cs=Cs, cv=cv, penalty="l2", solver=solver,
        scoring="accuracy", max_iter=2000, n_jobs=-1, refit=True
    )
    clf.fit(X_sub, y_sub)
    yhat = clf.predict(X_test_rep)
    return accuracy_score(y_test, yhat), float(clf.C_[0])

results = {"BoW": [], "LSA-CBOW": [], "GloVe-CBOW": []}
bestCs   = {"BoW": [], "LSA-CBOW": [], "GloVe-CBOW": []}

for n in sizes:
    idx = balanced_indices(y_train, n, seed=0)

    acc_bow,  C_bow  = cv_lr_test_acc(X_train[idx],       y_train[idx], X_test,       y_test, n)
    acc_lsa,  C_lsa  = cv_lr_test_acc(X_train_lsa[idx],   y_train[idx], X_test_lsa,   y_test, n)
    acc_glv,  C_glv  = cv_lr_test_acc(X_train_glove[idx], y_train[idx], X_test_glove, y_test, n)

    results["BoW"].append(acc_bow);        bestCs["BoW"].append(C_bow)
    results["LSA-CBOW"].append(acc_lsa);   bestCs["LSA-CBOW"].append(C_lsa)
    results["GloVe-CBOW"].append(acc_glv); bestCs["GloVe-CBOW"].append(C_glv)

# Print a compact table
print("\n=== 2.10 Learning Curves (test accuracy) ===")
for name in ["BoW", "LSA-CBOW", "GloVe-CBOW"]:
    accs = results[name]
    print(f"{name:10s}:", "  ".join(f"{n:>5d}:{a:.4f}" for n, a in zip(sizes, accs)))
print("\nBest C per size:")
for name in ["BoW", "LSA-CBOW", "GloVe-CBOW"]:
    Cs_str = "  ".join(f"{n:>5d}:{c:.4g}" for n, c in zip(sizes, bestCs[name]))
    print(f"{name:10s}: {Cs_str}")

# Plot and save
plt.figure(figsize=(6.2,4.2))
plt.plot(sizes, results["BoW"],        marker="o", label="BoW (10k)")
plt.plot(sizes, results["LSA-CBOW"],   marker="o", label="LSA-CBOW (10)")
plt.plot(sizes, results["GloVe-CBOW"], marker="o", label="GloVe-CBOW (300)")
plt.xscale("log")
plt.xticks(sizes, ["8","40","200","1K","5K","25K"])
plt.xlabel("Training examples (balanced)"); plt.ylabel("Test accuracy")
plt.title("Learning curves (4-fold CV LR)"); plt.grid(True, which="both", linestyle=":")
plt.legend(); plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
print("Saved plot -> learning_curves.png")






