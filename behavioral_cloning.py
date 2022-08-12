import glob
import os
import time
import zipfile
from typing import Iterator

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
    train = [[], [], 0, "train"]
    validation = [[], [], 0, "validation"]
    test = [[], [], 0, "test"]
    n_players = None
    for replay_path in sorted(glob.glob(f"{input_folder}/**/__game.parquet", recursive=True),
                              key=lambda p: os.path.basename(os.path.dirname(p))):
        replay_path = os.path.dirname(replay_path)
        replay_id = os.path.basename(replay_path)
        if int(replay_id[0], 16) < 3:
            continue

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
        # prev_actions = np.zeros((n_players, 8))
        # obs_builder = AdvancedObs()
        x_data, y_data = arrs[:2]
        for episode in label_replay(parsed_replay):
            df, actions = episode
            for obs, action in encoded_states_to_advanced_obs(df, actions):
                x_data.append(obs)
                y_data.append(action)
                arrs[2] += len(obs)

            # for i, (state, action) in enumerate(episode):
            #     if i == 0:
            #         obs_builder.reset(state)
            #         if (action == 8).all():
            #             continue
            #     for j, (player, act) in enumerate(zip(state.players, action)):
            #         obs = obs_builder.build_obs(player, state, prev_actions[j])
            #         x_data.append(obs)
            #         y_data.append(act)
            #         arrs[2] += 1
            #         if act.shape == (8,):
            #             prev_actions[j] = act
            #         else:
            #             prev_actions[j] = lookup_table[act.astype(int)]
        if arrs[2] > shard_size:
            x_data = np.concatenate(x_data)
            y_data = np.concatenate(y_data)
            assert len(x_data) == len(y_data)
            split = arrs[3]
            split_shards = sum(split in file for file in os.listdir(output_folder))
            np.savez_compressed(os.path.join(output_folder, f"{split}-shard-{split_shards}.npz"),
                                x_data=x_data, y_data=y_data)
            arrs[:3] = [], [], 0
        print(replay_id)
    for arrs in train, validation, test:
        x_data, y_data = arrs[:2]
        x_data = np.concatenate(x_data)
        y_data = np.concatenate(y_data)
        assert len(x_data) == len(y_data)
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

                    x_data = x_data[mask]
                    y_data = y_data[mask]
                except zipfile.BadZipFile:
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


def collate(batch):
    x, y = zip(*batch)
    # for b in batch:
    #     x.append(b[0])
    #     y.append(b[1])
    return (torch.from_numpy(np.stack(x)).float(),
            torch.from_numpy(np.stack(y)).long())


def train_bc():
    assert torch.cuda.is_available()
    train_dataset = BCDataset(r"D:\rokutleg\ssl-dataset-relabeled", "train")
    val_dataset = BCDataset(r"D:\rokutleg\ssl-dataset-relabeled", "validation", limit=10)

    ff_dim = 2048
    dropout_rate = 0.
    lr = 5e-5
    batch_size = 300

    model = BCNet(ff_dim, dropout_rate)
    print(model)
    model = torch.jit.trace(model, torch.zeros(10, 169)).cuda()
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e + 1))

    logger = wandb.init(group="behavioral-cloning", project="replay-model", entity="rolv-arild",
                        config=dict(ff_dim=ff_dim, dropout_rate=dropout_rate, lr=lr, batch_size=batch_size,
                                    optimizer=type(optimizer).__name__, lr_schedule=scheduler is not None))

    min_loss = float("inf")

    steps_per_hour = 108000
    assert steps_per_hour % batch_size == 0
    train_log_rate = 1
    val_log_rate = 24

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
                        torch.jit.save(model, f"bc-model-{logger.name}.pt")
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
    # import cProfile, pstats, io
    # from pstats import SortKey
    #
    # pr = cProfile.Profile()
    # pr.enable()

    # make_bc_dataset(r"E:\rokutleg\parsed\2021-ssl-replays\ranked-doubles",
    #                 r"D:\rokutleg\ssl-dataset-relabeled")

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
