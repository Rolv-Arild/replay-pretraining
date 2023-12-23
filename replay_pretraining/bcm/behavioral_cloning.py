import glob
import os
import time
import zipfile
from typing import Iterator, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.dataset import T_co

import wandb
from replays import load_parsed_replay, label_replay
from util import encoded_states_to_advanced_obs, lookup_table


def make_bc_dataset(input_folder, output_folder, shard_size=30 * 60 * 60):
    train = [[], [], 0, "train", []]
    validation = [[], [], 0, "validation", []]
    test = [[], [], 0, "test", []]
    n_players = None
    os.makedirs(output_folder, exist_ok=True)
    progress_file = open(os.path.join(output_folder, "_parsed_replays.txt"), "a+")
    progress_file.seek(0)
    parsed = set(progress_file.read().split("\n"))
    replay_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(input_folder) for f in fn if f == "__game.parquet"]
    for replay_path in sorted(replay_paths,
                              key=lambda p: os.path.basename(os.path.dirname(p))):
        replay_path = os.path.dirname(replay_path)
        replay_id = os.path.basename(replay_path)

        if replay_id in parsed:
            print(replay_id, "already parsed")
            continue

        s = sum(int(d, 16) for d in replay_id.replace("-", ""))
        if s % 100 < 96:
            arrs = train
        elif s % 100 < 98:
            arrs = validation
        else:
            arrs = test

        try:
            parsed_replay = load_parsed_replay(replay_path)
        except Exception as e:
            print("Error in replay", replay_id, e)
            continue
        if len(parsed_replay.metadata["players"]) % 2 != 0:
            continue
        elif n_players is None:
            n_players = len(parsed_replay.metadata["players"])
        elif len(parsed_replay.metadata["players"]) != n_players:
            continue

        try:
            x_data, y_data = arrs[:2]
            for episode in label_replay(parsed_replay):
                df, actions = episode
                for obs, action in encoded_states_to_advanced_obs(df, actions):
                    x_data.append(obs)
                    y_data.append(action)
                    arrs[2] += len(obs)
        except Exception as e:
            print("Error labeling replay", replay_id, e)
            continue

        arrs[4].append(replay_id)

        if arrs[2] > shard_size:
            x_data = np.concatenate(x_data)
            y_data = np.concatenate(y_data)
            assert len(x_data) == len(y_data)
            split = arrs[3]
            split_shards = sum(split in file for file in os.listdir(output_folder))
            np.savez_compressed(os.path.join(output_folder, f"{split}-shard-{split_shards}.npz"),
                                x_data=x_data, y_data=y_data)
            arrs[:3] = [], [], 0
            progress_file.write("\n".join(arrs[4]) + "\n")
            progress_file.flush()
            arrs[4] = []
        print(replay_id)
    for arrs in train, validation, test:
        x_data, y_data = arrs[:2]
        if len(x_data) == 0:
            continue
        x_data = np.concatenate(x_data)
        y_data = np.concatenate(y_data)
        assert len(x_data) == len(y_data)
        split = arrs[3]
        split_shards = sum(split in file for file in os.listdir(output_folder))
        np.savez_compressed(os.path.join(output_folder, f"{split}-shard-{split_shards}.npz"),
                            x_data=x_data, y_data=y_data)
        arrs[:3] = [], [], 0
        progress_file.write("\n".join(arrs[4]) + "\n")
        progress_file.flush()
        arrs[4] = []


class BCDataset(IterableDataset):
    def __init__(self, folder, key="train", limit=None):
        self.folder = folder
        self.key = key
        self.limit = limit

    def __iter__(self) -> Iterator[T_co]:
        n = 0

        action_shift = 0
        prev_actions_included = 0
        scorers_only = True

        while True:
            file = f"{self.key}-shard-{n}.npz"
            path = os.path.join(self.folder, file)
            if os.path.isfile(path):
                try:
                    f = np.load(path)
                    x_data = f["x_data"]
                    y_data = f["y_data"].astype(int)

                    mask = ~np.isnan(x_data).any(axis=-1)  # & (x_data[:, 1] == 0)

                    # Kickoffs tend to contain no-op at the start, making the model stand still when put in-game
                    mask &= ~((x_data[:, 0:2] == 0).all(axis=-1)  # Ball pos
                              & (x_data[:, 66:68] == 0).all(axis=-1)  # Player vel
                              & (lookup_table[y_data][:, 0] == 0))  # Action

                    # Physics data in replays tends to repeat 2 or 3 times, we only want value when it updates
                    # Players update independently, so let's only check data for the current player
                    mask &= (np.diff(x_data[:, 57:72], axis=0, prepend=np.nan) != 0).any(axis=1)

                    x_data = np.pad(
                        x_data,
                        ((0, 0),) * (len(x_data.shape) - 1)
                        + ((0, 231 - x_data.shape[-1] + 8 * prev_actions_included),)
                    )

                    if action_shift > 0:
                        x_data[:-action_shift, 9:17] = x_data[action_shift:, 9:17]

                    episode_ends = np.where(np.diff((x_data[:, 0:2] == 0).all(axis=-1).astype(int)) > 0)[0]
                    episode_start = 0
                    for episode_end in episode_ends.tolist() + [len(mask) - 1]:
                        if scorers_only and x_data[episode_end, 1] < 0:  # On agent's side of field
                            # mask[episode_start: episode_end + 1] = False
                            mask[max(episode_start, episode_end - 15 * 30): episode_end + 1] = False
                        for i in range(prev_actions_included):
                            actions_left = prev_actions_included - i
                            time_range_actions = slice(episode_start, max(episode_end - i, episode_start))
                            time_range_obs = slice(min(episode_start + i + 1, episode_end + 1), episode_end + 1)
                            action_range = slice(-8 * actions_left, -8 * (actions_left - 1) or None)
                            x_data[time_range_obs, action_range] = x_data[time_range_actions, 9:17]
                        mask[episode_end - action_shift + 1:episode_end + 1] = False
                        episode_start = episode_end + 1

                    if action_shift > 0:
                        x_data = x_data[:-action_shift]
                        y_data = y_data[action_shift:]
                        mask = mask[:-action_shift]

                    x_data = x_data[mask]
                    y_data = y_data[mask]
                except (zipfile.BadZipFile, EOFError):
                    n += 1
                    continue

                indices = np.random.permutation(len(x_data)) if self.key == "train" else np.arange(len(x_data))

                yield from zip(x_data[indices], y_data[indices])

                # if remainder is not None:
                #     x_data = np.concatenate((remainder[0], x_data))
                #     y_data = np.concatenate((remainder[1], y_data))
                #
                # indices = np.random.permutation(len(x_data)) if self.key == "train" else np.arange(len(x_data))
                # for b in range(0, len(indices), 1800):
                #     i = indices[b:b + 1800]
                #     if len(i) < 1800 and self.key == "train":
                #         remainder = x_data[i], y_data[i]
                #     else:
                #         yield x_data[i], y_data[i]
            else:
                # if remainder is not None and self.key != "train":
                #     yield remainder
                break
            n += 1
            if self.limit is not None and n >= self.limit:
                break


def add_relative_components(bba, oa):
    forward = bba[..., 60:63].unsqueeze(dim=-2)
    up = bba[..., 63:66].unsqueeze(dim=-2)
    left = torch.cross(up, forward)

    pitch = torch.arctan2(forward[..., 2], torch.sqrt(forward[..., 0] ** 2 + forward[..., 1] ** 2))
    yaw = torch.arctan2(forward[..., 1], forward[..., 0])
    roll = torch.arctan2(left[..., 2], up[..., 2])

    pitch = torch.unsqueeze(pitch, dim=-1)
    yaw = torch.unsqueeze(yaw, dim=-1)
    roll = torch.unsqueeze(roll, dim=-1)

    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    # Each of these holds 5 values for each player for each tick
    vals = torch.cat((oa[..., 1:7], oa[..., 10:16], oa[..., 19:22]), dim=-1)
    # vals[..., POS.start:LIN_VEL.stop] -= q[..., POS.start:LIN_VEL.stop]
    xs = vals[..., 0::3]
    ys = vals[..., 1::3]
    zs = vals[..., 2::3]

    # Rotation matrix with only yaw
    flip_relative_xs = cy * xs - sy * ys
    flip_relative_ys = sy * xs + cy * ys
    flip_relative_zs = zs

    # Now full rotation matrix
    car_relative_xs = cp * cy * xs + (sr * sp * cy - cr * sy) * ys - (cr * sp * cy + sr * sy) * zs
    car_relative_ys = cp * sy * xs + (sr * sp * sy + cr * cy) * ys - (cr * sp * sy - sr * cy) * zs
    car_relative_zs = sp * xs - cp * sr * ys + cp * cr * zs

    all_rows = torch.cat(
        (flip_relative_xs, flip_relative_ys, flip_relative_zs,
         car_relative_xs, car_relative_ys, car_relative_zs), dim=-1)

    return torch.cat((oa, all_rows), dim=-1)


def combined_pool(inp, mask=None, methods=("min", "max", "mean")):
    if mask is None:
        mask = (inp == 0).all(dim=-1)
    x = inp
    pooled = []

    # Multiply by 1e38 * 10 to produce inf where it is 1 and 0 otherwise, multiplying by inf causes nan at 0s
    a = torch.unsqueeze(mask * 1e38 * 1e38, -1)
    for method in methods:
        if method == "min":
            pooled.append(torch.min(x + a, dim=-2)[0])
        elif method == "max":
            pooled.append(torch.max(x - a, dim=-2)[0])
        elif method == "mean":
            pooled.append(torch.nanmean(x + (a - a), dim=-2))
        else:
            pooled.append(method(x, mask))
    x = torch.cat(pooled, dim=-1)
    return x


class ControlsPredictorDot(nn.Module):
    def __init__(self, in_features, features=32, layers=1, actions=None):
        super().__init__()
        if actions is None:
            actions = torch.from_numpy(lookup_table).float()
        else:
            actions = torch.from_numpy(actions).float()
        self.actions = nn.Parameter(actions)
        self.net = nn.Sequential(nn.Linear(8, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, features))  # Default 8->256->32
        self.emb_convertor = nn.Linear(in_features, features)

    def forward(self, player_emb: torch.Tensor, actions: Optional[torch.Tensor] = None):
        if actions is None:
            actions = self.actions
        player_emb = self.emb_convertor(player_emb)
        act_emb = self.net(actions)

        if act_emb.ndim == 2:
            return torch.einsum("ad,bd->ba", act_emb, player_emb)

        return torch.einsum("bad,bd->ba", act_emb, player_emb)


class BCNet(nn.Module):
    def __init__(self, input_dim, ff_dim, hidden_layers, dropout_rate):
        super().__init__()
        # self.conv = nn.Conv1d(45, conv_channels, kernel_size=(41,), stride=(1,))
        self.pre_bba = nn.ModuleList([nn.Linear(76, ff_dim)] +
                                     [nn.Linear(ff_dim, ff_dim) for _ in range(2)])
        self.pre_oa = nn.ModuleList([nn.Linear(32 + 30, ff_dim)] +
                                    [nn.Linear(ff_dim, ff_dim) for _ in range(2)])

        self.hidden_layers = nn.ModuleList([nn.Linear(4 * ff_dim, ff_dim)] +
                                           [nn.Linear(ff_dim, ff_dim) for _ in range(hidden_layers - 2)])
        self.dropout = nn.Dropout(dropout_rate)

        self.action_out = ControlsPredictorDot(ff_dim)

    def forward(self, inp: torch.Tensor):
        ball_boosts_agents = inp[..., :76]

        ball_boosts_agents[:, 9:14] = 0  # Throttle, steer, pitch, yaw, roll (too much bias)

        x_bba = ball_boosts_agents
        for layer in self.pre_bba:
            x_bba = layer(x_bba)
            x_bba = F.relu(self.dropout(x_bba))

        other_agents = inp[..., 76:]
        other_agents = other_agents.reshape(other_agents.shape[:-1] + (-1, 31))

        other_agents = F.pad(other_agents, (1, 0, 0, 5 - other_agents.shape[-2]))

        nonzero = (other_agents != 0).any(axis=-1)
        nz_cs = nonzero.cumsum(axis=-1)
        nz_s = nonzero.sum(axis=-1, keepdims=True)
        teammate_mask = nz_cs <= nz_s // 2
        other_agents[torch.where(teammate_mask) + (0,)] = 1

        x_oa = add_relative_components(ball_boosts_agents, other_agents)
        for layer in self.pre_oa:
            x_oa = layer(x_oa)
            x_oa = F.relu(self.dropout(x_oa))

        x_oa = combined_pool(x_oa, mask=~nonzero)

        x = torch.cat((x_bba, x_oa), dim=-1)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = F.relu(self.dropout(x))
        return self.action_out(x)


def collate(batch):
    x, y = zip(*batch)
    # for b in batch:
    #     x.append(b[0])
    #     y.append(b[1])
    # x = [np.pad(xs, pad_width=(0, 80 + 231 - xs.shape[0])) for xs in x]
    return (torch.from_numpy(np.stack(x)).float(),
            torch.from_numpy(np.stack(y)).long())


class CombinedDataset(IterableDataset):
    def __init__(self, *datasets: BCDataset):
        self.datasets = datasets

    def __iter__(self) -> Iterator[T_co]:
        iterators = [iter(d) for d in self.datasets]
        while True:
            try:
                items = [next(i) for i in iterators]
                yield from items
            except StopIteration:
                break


def train_bc():
    assert torch.cuda.is_available()
    # train_dataset = BCDataset(r"E:\rokutleg\ssl-dataset\ranked-doubles", "train")
    # val_dataset = BCDataset(r"E:\rokutleg\ssl-dataset\ranked-doubles", "validation", limit=10)

    train_dataset = CombinedDataset(
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-duels", "train"),
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-doubles", "train"),
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-standard", "train")
    )

    val_dataset = CombinedDataset(
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-duels", "validation", limit=10),
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-doubles", "validation", limit=10),
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-standard", "validation", limit=10)
    )

    ff_dim = 1024
    hidden_layers = 6
    dropout_rate = 0.
    lr = 5e-5
    batch_size = 300

    model = BCNet(231 + 0, ff_dim, hidden_layers, dropout_rate)
    print(model)
    j_inp = torch.normal(0, 1, (10, 231 + 0))
    j_inp[1, -(2 * 31 + 1)] = 0
    j_inp[2, -(4 * 31 + 1)] = 0
    model = torch.jit.trace(model, (j_inp,))  # , check_inputs=[(torch.cat((j_inp, j_inp), dim=0),)]).cuda()
    test = model(torch.cat((j_inp, j_inp), dim=0))
    test = model.cuda()(torch.cat((j_inp, j_inp), dim=0).cuda())
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 1 / (0.25 * e + 1))

    logger = wandb.init(group="behavioral-cloning", project="replay-model", entity="rolv-arild",
                        config=dict(ff_dim=ff_dim, dropout_rate=dropout_rate, lr=lr, batch_size=batch_size,
                                    optimizer=type(optimizer).__name__, lr_schedule=scheduler is not None))

    min_loss = float("inf")

    steps_per_hour = 108000
    assert steps_per_hour % batch_size == 0
    train_log_rate = 1
    val_log_rate = 7 * 24

    n = 0
    tot_loss = k = 0
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
                    loss = m = 0
                    correct = total = 0
                    for x_val, y_val in val_loader:
                        y_hat = model(x_val.cuda())
                        loss += loss_fn(y_hat, y_val.cuda()).item()
                        correct += (y_hat.cpu().argmax(axis=-1) == y_val).sum().item()
                        total += len(y_val)
                        m += 1
                    loss /= m
                    logger.log({"validation/loss": loss}, commit=False)
                    logger.log({"validation/accuracy": correct / total}, commit=False)
                    print(f"Day {total_samples // (val_log_rate * steps_per_hour)}:", loss)
                    if loss < min_loss:
                        torch.jit.save(torch.jit.trace(model, j_inp), f"bc-model-{logger.name}.pt")
                        print(f"Model saved at day {total_samples // (val_log_rate * steps_per_hour)} "
                              f"with total validation loss {loss}")
                        min_loss = loss
                model.train()

            y_hat = model(x_train.cuda())
            loss = loss_fn(y_hat, y_train.cuda())
            tot_loss += loss.item()
            k += 1
            if total_samples % (train_log_rate * steps_per_hour) == 0:
                logger.log({"epoch": epoch}, commit=False)
                logger.log({"train/samples": total_samples}, commit=False)
                logger.log({"train/loss": tot_loss / k}, commit=False)
                logger.log({"train/learning_rate": scheduler.get_last_lr()[0]})
                print(f"Hour {total_samples // steps_per_hour}:", tot_loss / k)
                tot_loss = k = 0

            loss.backward()
            optimizer.step()
            model.zero_grad(True)
            n += 1
        scheduler.step()


def test_bc():
    test_dataset = BCDataset(r"D:\rokutleg\ssl-dataset-fixed", "test")
    model = torch.jit.load("bc-model.pt").cuda()

    with open("../../bc-results.csv", "w") as f:
        f.write("action_pred,action_true\n")
        with torch.no_grad():
            model.eval()
            test_loader = DataLoader(test_dataset, 12, collate_fn=collate)
            for x, y_true in test_loader:
                y_hat = model(x.cuda())
                for j in range(len(x)):
                    f.write(f"{y_hat[j].argmax(axis=-1).item()},{y_true[j]}\n")


if __name__ == '__main__':
    # import cProfile, pstats, io
    # from pstats import SortKey
    #
    # pr = cProfile.Profile()
    # pr.enable()

    # make_bc_dataset(r"E:\rokutleg\parsed\2021-ssl-replays\ranked-doubles",
    #                 r"E:\rokutleg\ssl-dataset\ranked-doubles")

    # make_bc_dataset(r"E:\rokutleg\parsed\2021-ssl-replays\ranked-duels",
    #                 r"E:\rokutleg\ssl-dataset\ranked-duels")

    # make_bc_dataset(r"E:\rokutleg\parsed\2021-ssl-replays\ranked-standard",
    #                 r"E:\rokutleg\ssl-dataset\ranked-standard")

    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
    #
    # ps.dump_stats("profile_results.pstat")

    # time.sleep(3 * 60 * 60)
    train_bc()
    # test_bc()
