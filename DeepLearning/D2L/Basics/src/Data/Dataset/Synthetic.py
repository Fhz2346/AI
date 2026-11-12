import torch
import matplotlib.pyplot as plt
import random


def synthetic_data(w, b, n):
    X = torch.normal(0, 1, (n, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


def data_iter_self(batch_size, features, labels):
    n = len(features)
    indices = list(range(n))
    random.shuffle(indices)
    for i in range(0, n, batch_size):
        indices_batch = torch.tensor(indices[i : min(i + batch_size, n)])
        yield features[indices_batch], labels[indices_batch]


def visualize_data(features, labels):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        features[:, 0],
        features[:, 1],
        labels.reshape(-1),
        c=labels.reshape(-1),
        cmap="viridis",
        alpha=0.6,
    )
    plt.colorbar(scatter, label="Labels (z-axis)")
    ax.set_xlabel("Feature 1 (x-axis)")
    ax.set_ylabel("Feature 2 (y-axis)")
    ax.set_zlabel("Labels (z-axis)")
    ax.set_title("3D Visualization of Features and Labels")
    plt.show()
