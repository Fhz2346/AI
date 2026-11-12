import numpy as np
from tqdm import tqdm

from Train.Optimizer import SGD
from Network.Module.Loss import SoftmaxCrossEntropy
from Network.Module import Module
from Eval import evaluate


def data_iter(X_train, y_train, batch_size):
    num_examples = X_train.shape[0]
    indices = np.random.permutation(num_examples)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield X_train[batch_indices], y_train[batch_indices]


class TrainPipeline:

    def __init__(self, model: Module, cfg):
        self.model = model
        self.loss = SoftmaxCrossEntropy()
        self.optimizer = SGD(model.parameters(), lr=cfg.lr)
        self.batch_size = cfg.batch_size

    def train_epoch(self, X_train, y_train, batch_size):
        batch_losses = []
        for X_batch, y_batch in data_iter(X_train, y_train, batch_size):
            logits = self.model(X_batch)
            loss, dlogits = self.loss(logits, y_batch)

            self.optimizer.zero_grad()
            self.model.backward(dlogits)
            self.optimizer.step()
            batch_losses.append(loss)
        return float(np.mean(batch_losses))

    def train(self, X_train, y_train, X_test, y_test, num_epoch):
        pbar = tqdm(range(1, num_epoch + 1), desc="Training", unit="epoch")
        for t in pbar:
            loss = self.train_epoch(X_train, y_train, self.batch_size)
            err_train = evaluate(self.model, X_train, y_train)
            err_test = evaluate(self.model, X_test, y_test)
            pbar.set_postfix_str(
                f"iter/epoch {t:03d} | loss: {loss:.4f} | "
                f"train err: {err_train:.4f} | test err: {err_test:.4f}"
            )
