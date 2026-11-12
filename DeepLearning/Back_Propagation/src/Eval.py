import numpy as np

def evaluate(model, X, y):
    logits = model(X)
    pred = np.argmax(logits, axis=1)
    err = 1.0 - np.mean(pred == y)
    return err