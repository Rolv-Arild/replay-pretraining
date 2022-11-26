import os
import time
import zipfile
from typing import Iterator

import numba as numba
import numpy as np
import torch
from rlgym.utils.gamestates import GameState
from torch.optim.lr_scheduler import LambdaLR

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
    window_size = 38
    for file in os.listdir(in_folder):
        s = int(file.replace(".npz", ""), 13)
        if s % 100 < 90:
            arrs = train
        elif s % 100 < 95:
            arrs = validation
        else:
            arrs = test

        path = os.path.join(in_folder, file)
        f = np.load(path)
        states = f["states"]
        demos_md = np.diff(states[:, 86::39], axis=0) > 0
        demos_id = np.diff(states[:, 88::39], axis=0) > 0
        indices = np.where((~demos_id & demos_md))[0]
        pad = int(file.replace(".npz", "")) % 7 == 0
        if len(indices) > 0:
            states = states[:indices[0]]
            pad = False
        states = [GameState(arr.tolist()) for arr in states]
        actions = f["actions"]

        for x, y in get_data(states, actions):
            if not pad and len(x) < 2 * window_size + 1:
                continue
            window = rolling_window(np.arange(len(x)), 2 * window_size + 1, pad_start=pad, pad_end=pad)
            grouped_x = x[window]
            grouped_y = tuple(y[i][window[:, window_size]] for i in range(len(y)))
            normalize_quadrant(grouped_x, grouped_y)

            arrs[0].append((grouped_x, grouped_y))
            if sum(len(gx) for gx, gy in arrs[0]) > shard_size:
                x_data = np.concatenate([gx for gx, gy in arrs[0]])
                y_data = tuple(np.concatenate([gy[i] for gx, gy in arrs[0]])
                               for i in range(len(arrs[0][0][1])))
                out_name = f"{arrs[2]}-shard-{arrs[1]}.npz"
                print(f"{out_name} ({len(x_data)})")
                np.savez_compressed(os.path.join(out_folder, out_name), features=x_data, labels=y_data)
                arrs[1] += 1
                arrs[0].clear()


@numba.njit
def corrupted_indices(k, n):
    indices = np.zeros((k, n))
    for j in range(k):
        i = 0
        while i < n:
            r = np.random.random()
            if r < 0.075:
                repeats = 1
            elif r < 0.7:
                repeats = 2
            else:
                repeats = 3
            indices[j, i:i + repeats] = i
            i += repeats
    return indices


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
                try:
                    f = np.load(path)
                    features = f["features"]
                    labels = f["labels"]
                except zipfile.BadZipFile:
                    n += 1
                    continue

                # mask = ~np.isnan(features).any(axis=(1, 2))  # & (x_data[:, 1] == 0)
                #
                # features = features[mask]
                # labels = labels[:, mask]

                indices = (np.random.permutation(features.shape[0]) if self.key == "train"
                           else np.arange(features.shape[0]))

                features = features[indices]
                labels = labels[:, indices]

                if self.key == "train":
                    corrupt_mask = np.random.random(features.shape[0]) < 0.5
                    feat_corr = features[corrupt_mask]
                    ind = corrupted_indices(feat_corr.shape[0], feat_corr.shape[1]).astype(int)
                    feat_corr = feat_corr[np.tile(np.arange(feat_corr.shape[0]), (feat_corr.shape[1], 1)).T, ind]
                    features[corrupt_mask] = feat_corr

                yield from zip(features, zip(*labels))
            else:
                break
            n += 1
            if self.limit is not None and n >= self.limit:
                break


class IDMNet(nn.Module):
    def __init__(self, ff_dim, hidden_layers, dropout_rate):
        super().__init__()
        # self.conv = nn.Conv1d(45, conv_channels, kernel_size=(41,), stride=(1,))
        self.lin0 = nn.Linear(77 * 45, ff_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(ff_dim, ff_dim) for _ in range(hidden_layers)])
        self.dropout = nn.Dropout(dropout_rate)

        self.action_out = nn.Linear(ff_dim, 90)
        self.on_ground_out = nn.Linear(ff_dim, 2)
        self.has_jump_out = nn.Linear(ff_dim, 2)
        self.has_flip_out = nn.Linear(ff_dim, 2)

    def forward(self, x: torch.Tensor):
        # x = self.conv(x.swapaxes(1, 2))
        x = self.lin0(torch.reshape(x, (x.shape[0], 77 * 45)))
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


def collate(batch):
    x = []
    y = [[] for _ in range(4)]
    for b in batch:
        x.append(b[0])
        for i in range(len(y)):
            y[i].append(b[1][i])
    return (torch.from_numpy(np.stack(x)).float(),
            tuple(torch.from_numpy(np.stack(y[i])).long() for i in range(len(y))))


def train_idm():
    assert torch.cuda.is_available()
    output_names = ["action", "on_ground", "has_jump", "has_flip"]
    train_dataset = IDMDataset(r"D:\rokutleg\idm-dataset", "train")
    val_dataset = IDMDataset(r"D:\rokutleg\idm-dataset", "validation", limit=10)

    ff_dim = 1024
    hidden_layers = 6
    dropout_rate = 0.
    lr = 5e-5
    batch_size = 300

    model = IDMNet(ff_dim, hidden_layers, dropout_rate)
    print(model)
    model = torch.jit.trace(model, torch.zeros(10, 77, 45)).cuda()
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 1 / (0.2 * e + 1))

    logger = wandb.init(group="idm", project="replay-model", entity="rolv-arild",
                        config=dict(ff_dim=ff_dim, hidden_layers=hidden_layers,
                                    dropout_rate=dropout_rate, lr=lr, batch_size=batch_size,
                                    optimizer=type(optimizer).__name__, lr_schedule=scheduler is not None))

    min_loss = float("inf")

    steps_per_hour = 108000
    assert steps_per_hour % batch_size == 0
    train_log_rate = 1
    val_log_rate = 7 * 24

    n = 0

    train_losses = {k: 0 for k in output_names}
    train_inferences = 0
    train_correct = {k: 0 for k in output_names}
    train_samples = 0
    for epoch in range(100):
        model.train()
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate, drop_last=True)
        for x_train, y_train in train_loader:
            total_samples = batch_size * n

            if total_samples % (val_log_rate * steps_per_hour) == 0:
                print("Validating...")
                with torch.no_grad():
                    model.eval()
                    val_loader = DataLoader(val_dataset, 9000, collate_fn=collate)
                    val_losses = {k: 0 for k in output_names}
                    val_inferences = 0
                    val_correct = {k: 0 for k in output_names}
                    val_samples = 0
                    for x_val, y_val in val_loader:
                        y_hat = model(x_val.cuda())
                        for i, name in enumerate(output_names):
                            val_losses[name] += loss_fn(y_hat[i], y_val[i].cuda()).item()
                            val_correct[name] += (y_hat[i].cpu().argmax(axis=-1) == y_val[i]).sum().item()
                        val_samples += len(x_val)
                        val_inferences += 1
                    val_losses = {k: v / val_inferences for k, v in val_losses.items()}
                    accuracies = {k: v / val_samples for k, v in val_correct.items()}
                    loss = sum(val_losses.values())
                    logger.log({"validation/loss": loss}, commit=False)
                    logger.log({f"validation/{k}_loss": v for k, v in val_losses.items()}, commit=False)
                    logger.log({f"validation/{k}_accuracy": v for k, v in accuracies.items()}, commit=False)
                    print(f"Day {total_samples // (val_log_rate * steps_per_hour)}:", loss, val_losses)
                    if loss < min_loss:
                        torch.jit.save(model, f"idm-model-{logger.name}.pt")
                        print(f"Model saved at day {total_samples // (val_log_rate * steps_per_hour)} "
                              f"with total validation loss {loss}")
                        min_loss = loss
                model.train()

            y_hat = model(x_train.cuda())
            loss = 0
            for i, name in enumerate(output_names):
                l = loss_fn(y_hat[i], y_train[i].cuda())
                loss += l
                train_losses[name] += l.item()
                train_correct[name] += (y_hat[i].cpu().argmax(axis=-1) == y_train[i]).sum().item()
            train_inferences += 1
            train_samples += len(x_train)
            if total_samples % (train_log_rate * steps_per_hour) == 0:
                train_losses = {k: v / train_inferences for k, v in train_losses.items()}

                logger.log({"epoch": epoch}, commit=False)
                logger.log({"train/samples": total_samples}, commit=False)
                logger.log({"train/loss": sum(train_losses.values())}, commit=False)
                logger.log({f"train/{k}_loss": v for k, v in train_losses.items()}, commit=False)
                logger.log({f"train/{k}_accuracy": v / train_samples for k, v in train_correct.items()}, commit=False)
                logger.log({"train/learning_rate": scheduler.get_last_lr()[0]})
                print(f"Hour {total_samples // steps_per_hour}:", sum(train_losses.values()))
                train_losses = {k: 0 for k in output_names}
                train_inferences = 0
                train_correct = {k: 0 for k in output_names}
                train_samples = 0

            loss.backward()
            optimizer.step()
            model.zero_grad(True)
            n += 1
        scheduler.step()


def test_idm():
    output_names = ["action", "on_ground", "has_jump", "has_flip"]
    test_dataset = IDMDataset(r"D:\rokutleg\idm-dataset", "test")
    model = torch.jit.load("idm-model-super-star-16.pt").cuda()

    with open("idm-results-mc40.csv", "w") as f:
        f.write(",".join(f"{name}_{source}" for name in output_names for source in ("pred", "true")) + "\n")
        with torch.no_grad():
            # model.eval()
            test_loader = DataLoader(test_dataset, 12, collate_fn=collate)
            for x, y_true in test_loader:
                # y_hat = model(x.cuda())

                y_hat = [0.] * 4
                for _ in range(40):
                    t = model(x.cuda())
                    for j in range(4):
                        y_hat[j] += t[j]

                for j in range(len(x)):
                    s = ""
                    for i, name in enumerate(output_names):
                        s += f"{y_hat[i][j].argmax(axis=-1).item()},{y_true[i][j]},"
                    f.write(s[:-1] + "\n")


if __name__ == '__main__':
    # make_idm_dataset(r"D:\rokutleg\necto-data", r"D:\rokutleg\idm-dataset")
    train_idm()
    # test_idm()
