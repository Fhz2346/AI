import re
import numpy as np
import json
import hydra
from omegaconf import DictConfig

from Data.Vocab import Vocab
from Config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config", config_name="preprocess")
def main(cfg: DictConfig):
    with open(cfgG.data_dir / f"{cfg.dataset}.txt", "r") as f:
        lines = f.readlines()
    lines = [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]
    tokens = [list(line) for line in lines]
    vocab = Vocab(tokens, min_freq=cfg.min_freq)
    corpus = [vocab[token] for line in tokens for token in line]
    if cfg.max_tokens > 0:
        corpus = corpus[:cfg.max_tokens]
    np.save(cfgG.out_dir / "corpus.npy", np.array(corpus, dtype=np.int32))
    with open(cfgG.out_dir / "vocab.json", "w") as f:
        json.dump(vocab.idx2token, f)


if __name__ == "__main__":
    main()
