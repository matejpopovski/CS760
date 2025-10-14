import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    token_pattern=r"(?u)\b\w+\b",   # keep single-character tokens too
    max_features=MAX_VOCAB,
    binary=True                     # presence/absence (0/1)
)

#print("Vect: ", vect)

X_train = vect.fit_transform(train["text"])
#print("X_train: ", X_train)
X_test  = vect.transform(test["text"])   # IMPORTANT: transform with SAME vocab
#print("X_test: ", X_test)

print(f"X_train shape: {X_train.shape} (expected ~25k x 10k)")
print(f"X_test  shape: {X_test.shape}")

# -----------------------------
# 4) Train (unregularized) logistic regression and evaluate
# -----------------------------
# Preferred: truly unregularized if your sklearn supports it with lbfgs
try:
    clf = LogisticRegression(penalty="none", solver="lbfgs", max_iter=MAX_ITER)
    clf.fit(X_train, y_train)
except Exception:
    # Fallback matching the assignment hint ('liblinear'):
    # use very weak L2 (C large) to approximate no regularization
    clf = LogisticRegression(penalty="l2", C=1e9, solver="liblinear", max_iter=MAX_ITER)
    clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")




