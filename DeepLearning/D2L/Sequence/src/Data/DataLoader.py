import random
import torch


def seq_data_iter_random(corpus, batch_size, n_steps):
    corpus = corpus[random.randint(0, n_steps - 1) :]
    n_subSeqs = (len(corpus) - 1) // n_steps
    indices_init = list(range(0, n_subSeqs * n_steps, n_steps))
    random.shuffle(indices_init)

    def slice(pos):
        return corpus[pos : pos + n_steps]

    n_batches = n_subSeqs // batch_size
    for i in range(0, batch_size * n_batches, batch_size):
        indices_init_batch = indices_init[i : i + batch_size]
        X, Y = [], []
        for j in indices_init_batch:
            X.append(slice(j))
            Y.append(slice(j + 1))
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, n_step):
    offset = random.randint(0, n_step)
    n_token = ((len(corpus) - offset - 1) // batch_size) * batch_size

    Xs = corpus[offset : offset + n_token]
    Ys = corpus[offset + 1 : offset + n_token + 1]
    Xs = torch.tensor(Xs, dtype=torch.long).reshape(batch_size, -1)
    Ys = torch.tensor(Ys, dtype=torch.long).reshape(batch_size, -1)

    n_batch = Xs.shape[1] // n_step
    for i in range(0, n_step * n_batch, n_step):
        X = Xs[:, i : i + n_step]
        Y = Ys[:, i : i + n_step]
        yield X, Y


data_iter_dict = {
    "random": seq_data_iter_random,
    "sequential": seq_data_iter_sequential,
}


class SeqDataLoader:

    def __init__(self, corpus, cfg):
        self.batch_size = cfg.batch_size
        self.num_steps = cfg.num_steps
        
        self.corpus = corpus
        self.data_iter_fn = data_iter_dict[cfg.data_iter]

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
