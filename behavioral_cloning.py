import glob
import os
from typing import Iterator

import numpy as np
import torch
from rlgym.utils.obs_builders import AdvancedObs

import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.dataset import T_co

from replays import load_parsed_replay, label_replay, lut


def make_bc_dataset(input_folder, output_folder, shard_size=30 * 60 * 60):
    train = [[], [], 0, "train"]
    validation = [[], [], 0, "validation"]
    test = [[], [], 0, "test"]
    n_players = None
    for replay_path in sorted(glob.glob(f"{input_folder}/**/__game.parquet", recursive=True),
                              key=lambda p: os.path.basename(os.path.dirname(p))):
        replay_path = os.path.dirname(replay_path)
        replay_id = os.path.basename(replay_path)

        s = sum(int(d, 16) for d in replay_id.replace("-", ""))
        if s % 100 < 90:
            arrs = train
        elif s % 100 < 95:
            arrs = validation
        else:
            arrs = test

        try:
            parsed_replay = load_parsed_replay(replay_path)
        except Exception as e:
            print("Error in replay", replay_id, e)
            continue
        if n_players is None:
            n_players = len(parsed_replay.metadata["players"])
        elif len(parsed_replay.metadata["players"]) != n_players or n_players % 2 != 0:
            continue
        prev_actions = np.zeros((n_players, 8))
        obs_builders = [AdvancedObs() for _ in range(n_players)]
        x_data, y_data = arrs[:2]
        for episode in label_replay(parsed_replay):
            for i, (state, action) in enumerate(episode):
                for j, (player, obs_builder, act) in enumerate(zip(state.players, obs_builders, action)):
                    if i == 0:
                        obs_builder.reset(state)
                    obs = obs_builder.build_obs(player, state, prev_actions[j])
                    x_data.append(obs)
                    y_data.append(act)
                    arrs[2] += 1
                    if isinstance(act, int):
                        prev_actions[j] = lut[act]
                    else:
                        prev_actions[j] = act
        if arrs[2] > shard_size:
            x_data = np.stack(x_data)
            y_data = np.stack(y_data)
            split = arrs[3]
            split_shards = sum(split in file for file in os.listdir(output_folder))
            np.savez_compressed(os.path.join(output_folder, f"{split}-shard-{split_shards}.npz"),
                                x_data=x_data, y_data=y_data)
            arrs[:3] = [], [], 0
        print(replay_id)
    for arrs in train, validation, test:
        x_data, y_data = arrs[:2]
        x_data = np.stack(x_data)
        y_data = np.stack(y_data)
        split = arrs[3]
        split_shards = sum(split in file for file in os.listdir(output_folder))
        np.savez_compressed(os.path.join(output_folder, f"{split}-shard-{split_shards}.npz"),
                            x_data=x_data, y_data=y_data)
        arrs[:3] = [], [], 0


class BCDataset(IterableDataset):
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
                x_data = f["x_data"]
                mask = ~np.isnan(x_data).any(axis=-1)
                x_data = x_data[mask]
                y_data = f["y_data"][mask]
                assert mask.mean() > 0.5
                indices = np.random.permutation(len(x_data)) if self.key == "train" else np.arange(len(x_data))
                for b in range(0, len(indices), 1800):
                    i = indices[b:b + 1800]
                    yield x_data[i], y_data[i]
            else:
                break
            n += 1
            if self.limit is not None and n >= self.limit:
                break


class BCNet(nn.Module):
    def __init__(self, ff_dim, dropout_rate):
        super().__init__()
        # self.conv = nn.Conv1d(45, conv_channels, kernel_size=(41,), stride=(1,))
        self.lin0 = nn.Linear(169, ff_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(ff_dim, ff_dim) for _ in range(3)])
        self.dropout = nn.Dropout(dropout_rate)

        self.action_out = nn.Linear(ff_dim, 90)

    def forward(self, x: torch.Tensor):
        # x = self.conv(x.swapaxes(1, 2))
        x = self.lin0(x)
        x = F.relu(self.dropout(x))
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = F.relu(self.dropout(x))
        return self.action_out(x)


def train_bc():
    assert torch.cuda.is_available()
    train_dataset = BCDataset(r"D:\rokutleg\electrum-dataset", "train")
    val_dataset = BCDataset(r"D:\rokutleg\electrum-dataset", "validation", limit=4)

    ff_dim = 2048
    dropout_rate = 0.
    lr = 1e-4

    model = BCNet(ff_dim, dropout_rate)
    print(model)
    model = torch.jit.trace(model, torch.zeros(10, 169)).cuda()
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()

    logger = wandb.init(group="behavioral-cloning", project="replay-model", entity="rolv-arild",
                        config=dict(ff_dim=ff_dim, dropout_rate=dropout_rate, lr=lr))

    def collate(batch):
        x = []
        y = []
        for b in batch:
            x.append(b[0])
            y.append(b[1])
        return (torch.from_numpy(np.concatenate(x)).float(),
                torch.from_numpy(np.concatenate(y)).long())

    min_loss = float("inf")

    n = 0
    tot_loss = k = 0
    for epoch in range(100):
        model.train()
        train_loader = DataLoader(train_dataset, 5, collate_fn=collate)
        for x_train, y_train in train_loader:
            y_hat = model(x_train.cuda())
            loss = loss_fn(y_hat, y_train.cuda())
            tot_loss += loss.item()
            k += 1
            if n % 12 == 0:
                logger.log({"epoch": epoch}, commit=False)
                logger.log({"train/step": n * 1800 * 5}, commit=False)
                logger.log({"train/loss": tot_loss / k})
                print(f"Hour {n // 12}:", tot_loss / k)
                tot_loss = k = 0

            loss.backward()
            optimizer.step()
            model.zero_grad(True)
            n += 1

            if n % (12 * 24) == 0:
                print("Validating...")
                with torch.no_grad():
                    model.eval()
                    val_loader = DataLoader(val_dataset, 12, collate_fn=collate)
                    m = 0
                    loss = 0
                    for x_val, y_val in val_loader:
                        y_hat = model(x_val.cuda())
                        loss += loss_fn(y_hat, y_val.cuda())
                        m += 1
                    loss /= m
                    logger.log({"epoch": epoch}, commit=False)
                    logger.log({"validation/step": n * 1800 * 5}, commit=False)
                    logger.log({"validation/loss": loss.item()})
                    print(f"Day {n // (12 * 24)}:", loss.item())
                    if loss < min_loss:
                        torch.jit.save(model, "bc-model.pt")
                        print(f"Model saved at day {n // (12 * 24)} with total validation loss {loss}")
                        min_loss = loss


def test_bc():
    test_dataset = BCDataset(r"D:\rokutleg\electrum-dataset", "test")
    model = torch.jit.load("bc-model.pt").cuda()

    def collate(batch):
        x = []
        y = []
        for b in batch:
            x.append(b[0])
            y.append(b[1])
        return (torch.from_numpy(np.concatenate(x)).float(),
                torch.from_numpy(np.concatenate(y)).long())

    with open("bc-results.csv", "w") as f:
        f.write("action_pred,action_true\n")
        with torch.no_grad():
            model.eval()
            test_loader = DataLoader(test_dataset, 12, collate_fn=collate)
            for x, y_true in test_loader:
                y_hat = model(x.cuda())
                for j in range(len(x)):
                    f.write(f"{y_hat[j].argmax(axis=-1).item()},{y_true[j]}\n")
