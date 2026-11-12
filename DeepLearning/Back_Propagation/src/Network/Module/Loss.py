import numpy as np


class SoftmaxCrossEntropy:

    def __call__(self, logits, y):
        B, C = logits.shape
        y = y.astype(int).reshape(-1)

        m = np.max(logits, axis=1, keepdims=True)
        ex = np.exp(logits - m)
        probs = ex / np.sum(ex, axis=1, keepdims=True)

        p_y = probs[np.arange(B), y]
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0

        loss = -np.mean(np.log(p_y + 1e-12))
        dlogits /= B
        return loss, dlogits


class MSE:

    def __call__(self, pred, target):
        diff = pred - target
        loss = np.mean(diff * diff)
        dpred = (2.0 * diff) / diff.size
        return loss, dpred


class MAE:

    def __call__(self, pred, target):
        diff = pred - target
        loss = np.mean(np.abs(diff))
        dpred = np.sign(diff) / diff.size
        return loss, dpred


loss_dict = {
    "MSE": MSE,
    "MAE": MAE,
    "CE": SoftmaxCrossEntropy,
}
