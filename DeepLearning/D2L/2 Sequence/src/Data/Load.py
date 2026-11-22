import json
import numpy as np

from Data.Vocab import Vocab
from Data.DataLoader import SeqDataLoader
from Config import cfgG


def load_data_time_machine(cfg):
    vocab = Vocab()
    idx2token = json.load(open(cfgG.out_dir / "vocab.json", "r"))
    vocab.load_map_i2t(idx2token)
    corpus = np.load(cfgG.out_dir / "corpus.npy")
    data_iter = SeqDataLoader(corpus, cfg)
    return data_iter, vocab
