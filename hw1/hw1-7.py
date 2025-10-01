import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the data
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                x1, x2, label = map(float, parts)
                data.append([x1, x2, int(label)])
    return np.array(data)

# Custom Decision Tree Node
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold          # Threshold value for split
        self.left = left                    # Left child (<= threshold)
        self.right = right                  # Right child (> threshold)
        self.value = value                  # Value if leaf node

# Custom Decision Tree Classifier
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels"""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _best_split(self, X, y):
        """Find the best split for a node"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            # Get unique feature values and try thresholds between them
            feature_values = np.unique(X[:, feature_idx])
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2
            
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate weighted Gini impurity
                gini_left = self._gini_impurity(y[left_mask])
                gini_right = self._gini_impurity(y[right_mask])
                weighted_gini = (np.sum(left_mask) * gini_left + 
                               np.sum(right_mask) * gini_right) / n_samples
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gini
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return TreeNode(value=np.argmax(np.bincount(y)))
        
        # Find best split
        feature_idx, threshold, gini = self._best_split(X, y)
        
        if feature_idx is None:  # No good split found
            return TreeNode(value=np.argmax(np.bincount(y)))
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(feature_index=feature_idx, threshold=threshold,
                       left=left_subtree, right=right_subtree)
    
    def fit(self, X, y):
        """Train the decision tree"""
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x, node):
        """Predict a single sample"""
        if node.value is not None:  # Leaf node
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def count_nodes(self, node=None):
        """Count total number of nodes in the tree"""
        if node is None:
            node = self.root
        
        if node.value is not None:  # Leaf node
            return 1
        
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)

# Calculate zero-one loss (error rate)
def zero_one_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)

# Load data
print("Loading data...")
data = load_data('Dbig.txt')
X = data[:, :2]
y = data[:, 2].astype(int)

print(f"Total samples: {len(X)}")
print(f"Class distribution: {np.bincount(y)}")

# 1. Randomly split into training (8192) and test sets
n_total = len(X)
n_train = 8192
n_test = n_total - n_train

# Generate random permutation
rng = np.random.RandomState(42)  # for reproducibility
perm = rng.permutation(n_total)

# Split into training and test
train_idx = perm[:n_train]
test_idx = perm[n_train:]

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# 2. Create nested training sets
sizes = [32, 128, 512, 2048, 8192]
datasets = {}

for size in sizes:
    datasets[size] = {
        'X': X_train[:size],
        'y': y_train[:size]
    }

# 3. Train decision trees and evaluate
results = []

for size in sizes:
    print(f"Training tree with {size} samples...")
    
    # Get training data
    X_train_sub = datasets[size]['X']
    y_train_sub = datasets[size]['y']
    
    # Train custom decision tree
    clf = DecisionTree(max_depth=20, min_samples_split=2)
    clf.fit(X_train_sub, y_train_sub)
    
    # Make predictions on test set
    y_pred = clf.predict(X_test)
    
    # Calculate error
    error = zero_one_loss(y_test, y_pred)
    
    # Get number of nodes
    n_nodes = clf.count_nodes()
    
    results.append({
        'n': size,
        'nodes': n_nodes,
        'error': error,
        'classifier': clf
    })
    
    print(f"n={size}: {n_nodes} nodes, error={error:.4f}")

# 4. Create the three-column table
print("\n" + "="*50)
print("THREE-COLUMN TABLE:")
print("="*50)
print("n\tNodes\tError")
print("-" * 25)
for result in results:
    print(f"{result['n']}\t{result['nodes']}\t{result['error']:.4f}")

# 5. Plot learning curve
plt.figure(figsize=(10, 6))
n_values = [r['n'] for r in results]
errors = [r['error'] for r in results]
nodes = [r['nodes'] for r in results]

plt.plot(n_values, errors, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Training Set Size (n)')
plt.ylabel('Test Error')
plt.title('Learning Curve: Training Set Size vs Test Error')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xticks(n_values, n_values)
plt.tight_layout()
plt.savefig('learning_curve_custom.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Visualize decision boundaries
def plot_decision_boundary(clf, X, y, title, ax):
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict on mesh grid (this might be slow for our custom implementation)
    print(f"Generating decision boundary for {title}...")
    Z = np.array([clf._predict_single(np.array([x, y]), clf.root) 
                  for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    
    # Plot training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, 
                        edgecolors='black', alpha=0.6, s=20)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    return scatter

# Create subplots for decision boundaries
print("\nGenerating decision boundary plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

# Remove the last subplot since we only have 5 trees
fig.delaxes(axes[5])

for i, result in enumerate(results):
    n = result['n']
    plot_decision_boundary(result['classifier'], 
                          datasets[n]['X'], 
                          datasets[n]['y'],
                          f'n={n}\nNodes: {result["nodes"]}, Error: {result["error"]:.3f}',
                          axes[i])

plt.tight_layout()
plt.savefig('decision_boundaries_custom.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional analysis: Plot nodes vs error
plt.figure(figsize=(10, 6))
plt.plot(nodes, errors, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Nodes in Tree')
plt.ylabel('Test Error')
plt.title('Tree Complexity vs Test Error')
plt.grid(True, alpha=0.3)
for i, (n, node, err) in enumerate(zip(n_values, nodes, errors)):
    plt.annotate(f'n={n}', (node, err), xytext=(5, 5), textcoords='offset points')
plt.tight_layout()
plt.savefig('complexity_vs_error.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary
print("\n" + "="*50)
print("SUMMARY:")
print("="*50)
print("As training set size increases:")
print("- Test error generally decreases (learning occurs)")
print("- Number of nodes in the tree increases")
print("- Decision boundaries become more complex and detailed")
print("- The model captures more intricate patterns in the data")

print(f"\nFinal results:")
for result in results:
    print(f"n={result['n']:4d}: {result['nodes']:3d} nodes, error={result['error']:.4f}")