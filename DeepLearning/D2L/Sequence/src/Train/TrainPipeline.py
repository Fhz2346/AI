import torch
from torch import nn
import torch.nn.functional as F
import math
from tqdm import tqdm

from Train.Utils import grad_clipping
from Utils.Auxiliary import Timer, Accumulator
from Model.RNN import RNN
from Model.RNN_dict import RNN_dict
from Model.SequencePredictor import SequencePredictor
from Config import cfgG


class TrainPipeline:

    def __init__(self, cfg, vocab, device):
        self.vocab = vocab
        self.device = device
        vocab_size = len(vocab)
        rnn = RNN_dict[cfg.network.type](dI=vocab_size, dH=cfg.network.dH, device=device)
        self.net = SequencePredictor(rnn, vocab_size)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), cfg.train.lr)

        self.use_random_iter = cfg.train.data.data_iter == "random"

    def process_state(self, state):
        if state is None or self.use_random_iter:
            state = self.net.rnn.begin_state(batch_size=1)
        else:
            if isinstance(state, tuple):
                for s in state:
                    s.detach_()
            elif isinstance(state, torch.Tensor):
                state.detach_()
            else:
                raise TypeError("Unknown state")
        return state

    def train_epoch(self, train_iter):
        timer = Timer()
        metric = Accumulator(2)
        vocab_size = len(self.vocab)
        state = None
        for X, Y in train_iter:
            state = self.process_state(state)  # detach state every num_steps
            X = F.one_hot(X.T, vocab_size).float().to(self.device)
            Y_hat, state = self.net(X, state)

            Y_flat = Y.T.reshape(-1).to(self.device)
            Y_hat_flat = Y_hat.reshape(-1, Y_hat.shape[-1])
            l = self.loss(Y_hat_flat, Y_flat).mean()

            self.optimizer.zero_grad()
            l.backward()
            grad_clipping(self.net, 1)
            self.optimizer.step()

            metric.add(l * Y_flat.numel(), Y_flat.numel())
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

    def train(self, train_iter, num_epochs, save_freq):
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            ppl, speed = self.train_epoch(train_iter)
            if (epoch + 1) % 10 == 0:
                pbar.set_description(f"epoch {epoch + 1}, 困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(self.device)}")
            if (epoch + 1) % save_freq == 0:
                save_path = cfgG.checkpoint_dir / self.net.rnn.__class__.__name__ / f"{epoch+1}.pth"
                torch.save(self.net.state_dict(), save_path)
