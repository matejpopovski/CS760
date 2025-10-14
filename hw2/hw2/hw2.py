# -*- coding: utf-8 -*-
# CS760 IMDB HW — Sections 2.1 and 2.3 in one file

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

# 5) (Optional) save for later use
# np.save("R_lsa_tokens_10d.npy", R_lsa)
