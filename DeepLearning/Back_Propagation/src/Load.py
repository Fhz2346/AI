from sklearn.datasets import fetch_openml
import numpy as np
from Config import cfgG

def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, data_home=cfgG.data_dir)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)
    X /= 255.0
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
