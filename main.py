import os
from typing import Iterator, List

import numpy as np
import torch
import wandb
from rlgym.utils.common_values import BALL_RADIUS
from rlgym.utils.gamestates.game_state import GameState
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co, IterableDataset
import torch.nn.functional as F


def make_lookup_table():
    actions = []
    # Ground
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    # Aerial
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:  # Only need roll for sideflip
                            continue
                        if pitch == roll == jump == 0:  # Duplicate with ground
                            continue
                        # Enable handbrake for potential wavedashes
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    actions = np.array(actions)
    return actions


LIN_NORM = 1 / 2300
ANG_NORM = 1 / 5.5
lookup_table = make_lookup_table()

mirror_map = []
m_action = np.array([1, -1, 1, -1, -1, 1, 1, 1])
for action in lookup_table * m_action:
    i = np.where((lookup_table == action).all(axis=-1))[0][0]
    mirror_map.append(i)
mirror_map = np.array(mirror_map, dtype=int)


def normalize_quadrant(features, label):
    actions = label[0]
    mid = features.shape[1] // 2
    neg_x = features[:, mid, 0] < 0
    neg_y = features[:, mid, 1] < 0
    mirrored = neg_x ^ neg_y  # Only one of them

    lin_cols = np.r_[0:3, 3:6, 6:9, 9:12, 18:21, 21:24, 29:32, 32:35, 35:38, 38:41]
    ang_cols = np.r_[12:15, 24:27, 41:44]

    transform = np.ones(features.shape[-1])
    transform[lin_cols] = np.tile(np.array([-1, 1, 1]), len(lin_cols) // 3)
    features[neg_x] *= transform

    transform[:] = 1
    transform[ang_cols] = np.tile(np.array([1, -1, -1]), len(ang_cols) // 3)
    features[neg_x] *= transform

    transform[:] = 1
    transform[lin_cols] = np.tile(np.array([1, -1, 1]), len(lin_cols) // 3)
    features[neg_y] *= transform

    transform[:] = 1
    transform[ang_cols] = np.tile(np.array([-1, 1, -1]), len(ang_cols) // 3)
    features[neg_y] *= transform

    actions[mirrored] = mirror_map[actions[mirrored].astype(int)]

    return mirrored


def get_data(states: List["GameState"], actions: np.ndarray):
    positions = np.array([[p.car_data.position for p in state.players] for state in states])
    for i in range(len(states[0].players)):
        x_data = np.zeros((len(states), 45))
        y_data = tuple(np.zeros(len(states)) for _ in range(4))
        for j, state in enumerate(states):
            player = state.players[i]
            action = actions[j][i]

            features = np.zeros(45)
            features[0:3] = player.car_data.position * LIN_NORM
            features[3:6] = player.car_data.linear_velocity * LIN_NORM
            features[6:9] = player.car_data.forward()
            features[9:12] = player.car_data.up()
            features[12:15] = player.car_data.angular_velocity * ANG_NORM
            features[15] = player.boost_amount
            features[16] = player.is_demoed

            if np.linalg.norm(state.ball.position - player.car_data.position) < 4 * BALL_RADIUS:
                features[17] = 1
                features[18:21] = state.ball.position * LIN_NORM
                features[21:24] = state.ball.linear_velocity * LIN_NORM
                features[24:27] = state.ball.angular_velocity * ANG_NORM

            dists = np.linalg.norm(positions[j] - player.car_data.position, axis=-1)
            closest = np.argsort(dists)[1]
            if dists[closest] < 3 * BALL_RADIUS:
                p = state.players[closest]
                features[27] = 1
                features[28] = p.team_num == player.team_num
                features[29:32] = p.car_data.position * LIN_NORM
                features[32:35] = p.car_data.linear_velocity * LIN_NORM
                features[35:38] = p.car_data.forward()
                features[38:41] = p.car_data.up()
                features[41:44] = p.car_data.angular_velocity * ANG_NORM
                features[44] = p.boost_amount

            x_data[j] = features
            y_data[0][j] = action
            y_data[1][j] = player.on_ground
            y_data[2][j] = player.has_jump
            y_data[3][j] = player.has_flip
        yield x_data, y_data


def rolling_window(a, window, pad_start=False, pad_end=False):
    # https://stackoverflow.com/questions/29875687/numpy-grouping-every-n-continuous-element
    if pad_start:
        a = np.concatenate((np.array(window // 2 * [a[0]]), a))
    if pad_end:
        a = np.concatenate((a, np.array(window // 2 * [a[-1]])))

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def make_dataset(in_folder, out_folder, shard_size=60 * 60 * 30):
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


if __name__ == '__main__':
    # make_dataset(r"D:\rokutleg\necto-data", r"D:\rokutleg\idm-dataset")
    # train_idm()
    # test_idm()
    train_bc()
    # test_bc()
    # for i, (state, actions) in enumerate(to_rlgym(load_parsed_replay("example_parsed_replay"))):
    #     if np.isnan(state.ball.position).any():
    #         print("1234")
    #     if state.boost_pads.any():
    #         print(i, np.where(state.boost_pads > 0)[0].tolist())
