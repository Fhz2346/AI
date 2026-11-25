import torch.nn as nn
from torch.optim import SGD
from .dataset import *
from .resnet import RESNET

model = RESNET().to(device=DEV, dtype=model_dt)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=5e-4)
assert MODEL_DT == dtype2name[next(model.parameters()).dtype]


def train_step(it):
    x, y = next(it)
    x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    return loss
