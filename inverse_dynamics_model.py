import os
from typing import Iterator

import numpy as np
import torch
from rlgym.utils.gamestates import GameState

import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.dataset import T_co

from util import get_data, rolling_window, normalize_quadrant


def make_idm_dataset(in_folder, out_folder, shard_size=60 * 60 * 30):
    train = [[], 0, "train"]
    validation = [[], 0, "validation"]
    test = [[], 0, "test"]
    for file in os.listdir(in_folder):
        path = os.path.join(in_folder, file)
        f = np.load(path)
        states = [GameState(arr.tolist()) for arr in f["states"]]
        actions = f["actions"]

        s = sum(int(d) for d in file.replace(".npz", ""))
        if s % 10 < 8:
            arrs = train
        elif s % 10 == 8:
            arrs = validation
        else:
            arrs = test

        for x, y in get_data(states, actions):
            if len(x) < 41:
                continue
            window = rolling_window(np.arange(len(x)), 41)
            grouped_x = x[window]
            grouped_y = tuple(y[i][window[:, 20]] for i in range(len(y)))
            normalize_quadrant(grouped_x, grouped_y)

            arrs[0].append((grouped_x, grouped_y))
            if sum(len(gx) for gx, gy in arrs[0]) > shard_size:
                x_data = np.concatenate([gx for gx, gy in arrs[0]])
                y_data = tuple(np.concatenate([gy[i] for gx, gy in arrs[0]])
                               for i in range(len(arrs[0][0][1])))
                out_name = f"{arrs[2]}-shard-{arrs[1]}.npz"
                print(f"{out_name} ({len(arrs[0])})")
                np.savez_compressed(os.path.join(out_folder, out_name), features=x_data, labels=y_data)
                arrs[1] += 1
                arrs[0].clear()


class IDMDataset(IterableDataset):
    def __init__(self, folder, key="train", limit=None):
        self.folder = folder
        self.key = key
        self.limit = limit

    def __iter__(self) -> Iterator[T_co]:
        n = 0
        while True:
            file = f"{self.key}-shard-{n}.npz"
            path = os.path.join(self.folder, file)
            if os.path.isfile(path):
                f = np.load(path)
                features = f["features"]
                labels = f["labels"]
                indices = np.random.permutation(len(features)) if self.key == "train" else np.arange(len(features))
                for b in range(0, len(indices), 1800):
                    i = indices[b:b + 1800]
                    yield features[i], tuple(labels[j][i] for j in range(len(labels)))
            else:
                break
            n += 1
            if self.limit is not None and n >= self.limit:
                break


class IDMNet(nn.Module):
    def __init__(self, ff_dim, dropout_rate):
        super().__init__()
        # self.conv = nn.Conv1d(45, conv_channels, kernel_size=(41,), stride=(1,))
        self.lin0 = nn.Linear(1845, ff_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(ff_dim, ff_dim) for _ in range(3)])
        self.dropout = nn.Dropout(dropout_rate)

        self.action_out = nn.Linear(ff_dim, 90)
        self.on_ground_out = nn.Linear(ff_dim, 2)
        self.has_jump_out = nn.Linear(ff_dim, 2)
        self.has_flip_out = nn.Linear(ff_dim, 2)

    def forward(self, x: torch.Tensor):
        # x = self.conv(x.swapaxes(1, 2))
        x = self.lin0(torch.reshape(x, (x.shape[0], 1845)))
        x = F.relu(self.dropout(x))
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = F.relu(self.dropout(x))
        return (
            self.action_out(x),
            self.on_ground_out(x),
            self.has_jump_out(x),
            self.has_flip_out(x)
        )


def train_idm():
    assert torch.cuda.is_available()
    output_names = ["action", "on_ground", "has_jump", "has_flip"]
    train_dataset = IDMDataset(r"D:\rokutleg\idm-dataset", "train")
    val_dataset = IDMDataset(r"D:\rokutleg\idm-dataset", "validation", limit=4)

    ff_dim = 2048
    dropout_rate = 0.5
    lr = 5e-5

    model = IDMNet(ff_dim, dropout_rate)
    print(model)
    model = torch.jit.trace(model, torch.zeros(10, 41, 45)).cuda()
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()

    logger = wandb.init(group="idm", project="replay-model", entity="rolv-arild",
                        config=dict(ff_dim=ff_dim, dropout_rate=dropout_rate, lr=lr))

    def collate(batch):
        x = []
        y = [[] for _ in range(4)]
        for b in batch:
            x.append(b[0])
            for i in range(len(y)):
                y[i].append(b[1][i])
        return (torch.from_numpy(np.concatenate(x)).float(),
                tuple(torch.from_numpy(np.concatenate(y[i])).long() for i in range(len(y))))

    min_loss = float("inf")

    n = 0
    for epoch in range(100):
        model.train()
        train_loader = DataLoader(train_dataset, 5, collate_fn=collate)
        for x_train, y_train in train_loader:
            y_hat = model(x_train.cuda())
            losses = {}
            for i, name in enumerate(output_names):
                losses[name] = loss_fn(y_hat[i].squeeze(), y_train[i].cuda())
            loss = sum(losses.values())
            if n % 12 == 0:
                logger.log({"epoch": epoch}, commit=False)
                logger.log({"train/step": n * 1800 * 5}, commit=False)
                logger.log({"train/total_loss": loss.item()}, commit=False)
                logger.log({f"train/{k}_loss": v.item() for k, v in losses.items()})
                print(f"Hour {n // 12}:", loss.item(), {k: v.item() for k, v in losses.items()})

            loss.backward()
            optimizer.step()
            model.zero_grad(True)
            n += 1

            if n % (12 * 24) == 0:
                print("Validating...")
                with torch.no_grad():
                    model.eval()
                    val_loader = DataLoader(val_dataset, 12, collate_fn=collate)
                    losses = {k: 0 for k in output_names}
                    m = 0
                    for x_val, y_val in val_loader:
                        y_hat = model(x_val.cuda())
                        for i, name in enumerate(output_names):
                            losses[name] += loss_fn(y_hat[i].squeeze(), y_val[i].cuda())
                        m += 1
                    tot_loss = sum(losses.values()).item() / m
                    logger.log({"epoch": epoch}, commit=False)
                    logger.log({"validation/step": n * 1800 * 5}, commit=False)
                    logger.log({"validation/total_loss": tot_loss}, commit=False)
                    logger.log({f"validation/{k}_loss": v.item() / m for k, v in losses.items()})
                    print(f"Day {n // (12 * 24)}:", tot_loss, {k: v.item() / m for k, v in losses.items()})
                    if tot_loss < min_loss:
                        torch.jit.save(model, "idm-model.pt")
                        print(f"Model saved at day {n // (12 * 24)} with total validation loss {tot_loss}")
                        min_loss = tot_loss


def test_idm():
    output_names = ["action", "on_ground", "has_jump", "has_flip"]
    test_dataset = IDMDataset(r"D:\rokutleg\idm-dataset", "test")
    model = torch.jit.load("idm-model.pt").cuda()

    def collate(batch):
        x = []
        y = [[] for _ in range(4)]
        for b in batch:
            x.append(b[0])
            for i in range(len(y)):
                y[i].append(b[1][i])
        return (torch.from_numpy(np.concatenate(x)).float(),
                tuple(torch.from_numpy(np.concatenate(y[i])).long() for i in range(len(y))))

    with open("idm-results.csv", "w") as f:
        f.write(",".join(f"{name}_{source}" for name in output_names for source in ("pred", "true")) + "\n")
        with torch.no_grad():
            model.eval()
            test_loader = DataLoader(test_dataset, 12, collate_fn=collate)
            for x, y_true in test_loader:
                y_hat = model(x.cuda())
                for j in range(len(x)):
                    s = ""
                    for i, name in enumerate(output_names):
                        s += f"{y_hat[i][j].argmax(axis=-1).item()},{y_true[i][j]},"
                    f.write(s[:-1] + "\n")
