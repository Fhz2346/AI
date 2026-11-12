class SGD:
    def __init__(self, params, lr=1e-2, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        for p in self.params:
            if self.weight_decay != 0.0:
                p.grad += self.weight_decay * p.data
            p.data -= self.lr * p.grad
