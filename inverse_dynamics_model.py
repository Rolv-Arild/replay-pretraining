import os
import random
import time
import zipfile
from hashlib import md5
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

from util import get_data, rolling_window, normalize_quadrant, ControlsPredictorDot, ControlsPredictorLinear, \
    random_action, mutate_action


def make_idm_dataset(in_folder, out_folder, shard_size=60 * 60 * 30):
    train = [[], 0, "train"]
    validation = [[], 0, "validation"]
    test = [[], 0, "test"]
    window_size = 38
    for file in os.listdir(in_folder):
        file_idx = int(file.replace("states-actions-episodes-", "").replace(".npz", ""))

        path = os.path.join(in_folder, file)
        f = np.load(path)
        all_states = f["states"]
        all_episodes = f["episodes"]
        all_actions = f["actions"].reshape(-1, 6, 8)
        for episode in np.unique(all_episodes):
            mask = all_episodes == episode

            s = hash(f"{file}{episode}")

            if s % 100 < 90:
                arrs = train
            elif s % 100 < 95:
                arrs = validation
            else:
                arrs = test

            states = all_states[mask]

            pad = False  # episode % 7 == 0

            states = [GameState(arr.tolist()) for arr in states]
            actions = all_actions[mask]

            for x, y in get_data(states, actions):
                if not pad and len(x) < 2 * window_size + 1:
                    continue
                window = rolling_window(np.arange(len(x)), 2 * window_size + 1, pad_start=pad, pad_end=pad)
                grouped_x = x[window]
                grouped_y = tuple(y[i][window[:, window_size]] for i in range(len(y)))
                normalize_quadrant(grouped_x, grouped_y)
                assert grouped_y[0].max() <= 1

                arrs[0].append((grouped_x, grouped_y))
                if sum(len(gx) for gx, gy in arrs[0]) > shard_size:
                    x_data = np.concatenate([gx for gx, gy in arrs[0]])
                    y_data = tuple(np.concatenate([gy[i] for gx, gy in arrs[0]])
                                   for i in range(len(arrs[0][0][1])))
                    out_name = f"{arrs[2]}-shard-{arrs[1]}.npz"
                    print(f"{out_name} ({len(x_data)})")
                    np.savez_compressed(os.path.join(out_folder, out_name), features=x_data,
                                        actions=y_data[0], random_actions=y_data[1], mutated_actions=y_data[2],
                                        on_ground=y_data[3], has_jump=y_data[4], has_flip=y_data[5])
                    arrs[1] += 1
                    arrs[0].clear()


@numba.njit
def corrupted_indices(k, n):
    indices = np.zeros((k, n))
    for j in range(k):
        i = 0
        while i < n:
            r = np.random.random()
            if r < 0.15:
                repeats = 1  # 15%
            elif r < 0.65:
                repeats = 2  # 50%
            else:
                repeats = 3  # 35%
            indices[j, i:i + repeats] = i
            i += repeats
    return indices


class IDMDataset(IterableDataset):
    def __init__(self, folder, key="train", limit=None, num_action_options=10, corrupt=True, use_center_diff=True,
                 mutate_action=False, shuffle=True):
        self.folder = folder
        self.key = key
        self.limit = limit
        self.num_action_options = num_action_options
        self.corrupt = corrupt
        self.use_center_diff = use_center_diff
        self.mutate_action = mutate_action
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[T_co]:
        files = [file for file in os.listdir(self.folder) if self.key in file]
        files = sorted(files, key=lambda x: int(x.split("-")[-1].split(".")[0]))
        files = files[:self.limit]

        random.shuffle(files)
        for file in files:
            path = os.path.join(self.folder, file)

            rng = np.random if self.shuffle else np.random.RandomState(int(file.split("-")[-1].split(".")[0]))

            try:
                f = np.load(path)

                features = f["features"]
                actions = f["actions"]
                random_actions = f["random_actions"]
                mutated_actions = f["mutated_actions"]
                on_ground = f["on_ground"]
                has_jump = f["has_jump"]
                has_flip = f["has_flip"]
            except zipfile.BadZipFile:
                continue

            # print(f"Stats (n={n})")
            # print(features.min(), features.mean(), features.max())
            # print(actions.min(), actions.mean(), actions.max())
            # print(on_ground.min(), on_ground.mean(), on_ground.max())
            # print(has_jump.min(), has_jump.mean(), has_jump.max())
            # print(has_flip.min(), has_flip.mean(), has_flip.max())

            indices = rng.permutation(features.shape[0]) if self.shuffle else np.arange(features.shape[0])

            features = features[indices]
            actions = actions[indices]
            on_ground = on_ground[indices]
            has_jump = has_jump[indices]
            has_flip = has_flip[indices]

            action_options = np.repeat(np.expand_dims(actions, axis=1), self.num_action_options, axis=1)
            action_indices = np.zeros(action_options.shape[0])
            action_options[:, 0] = actions
            action_population = mutated_actions if self.mutate_action else random_actions

            if self.shuffle:
                # There might be more available actions than num_action_options, so we select a random subset for each
                action_population = action_population.swapaxes(0, 1)
                rng.shuffle(action_population)
                action_population = action_population.swapaxes(0, 1)
                action_options[:, 1:] = action_population[:, :self.num_action_options - 1]
            else:
                action_options[:, 1:] = action_population[:, :self.num_action_options - 1]

            if self.corrupt:
                corrupt_mask = rng.random(features.shape[0]) < 0.5
                feat_corr = features[corrupt_mask]
                ind = corrupted_indices(feat_corr.shape[0], feat_corr.shape[1]).astype(int)
                feat_corr = feat_corr[np.tile(np.arange(feat_corr.shape[0]), (feat_corr.shape[1], 1)).T, ind]
                features[corrupt_mask] = feat_corr

            if self.use_center_diff:
                # Make features the difference from the central frame
                mid = features.shape[1] // 2
                features[:, np.r_[0:mid, mid + 1:features.shape[1]]] -= features[:, mid:mid + 1]

            for i in range(features.shape[0]):
                yield (features[i][..., :16], action_options[i]), (
                    action_indices[i], on_ground[i], has_jump[i], has_flip[i])


class IDMNet(nn.Module):
    def __init__(self, ff_dim, hidden_layers, dropout_rate):
        super().__init__()
        # self.conv = nn.Conv1d(45, conv_channels, kernel_size=(41,), stride=(1,))
        self.lin0 = nn.Linear(77 * 16, ff_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(ff_dim, ff_dim) for _ in range(hidden_layers)])
        self.dropout = nn.Dropout(dropout_rate)

        self.action_out = ControlsPredictorDot(ff_dim, ff_dim, ff_dim // 16, 2)
        self.on_ground_out = nn.Linear(ff_dim, 2)
        self.has_jump_out = nn.Linear(ff_dim, 2)
        self.has_flip_out = nn.Linear(ff_dim, 2)

    def forward(self, x: torch.Tensor, action_options: torch.Tensor):
        # x = self.conv(x.swapaxes(1, 2))
        x = self.lin0(torch.reshape(x, (x.shape[0], 77 * 16)))
        x = F.gelu(self.dropout(x))
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = F.gelu(self.dropout(x))
        return (
            self.action_out(x, action_options),
            self.on_ground_out(x),
            self.has_jump_out(x),
            self.has_flip_out(x)
        )


def collate(batch):
    x = [[] for _ in range(2)]
    y = [[] for _ in range(4)]
    for b in batch:
        for i in range(len(x)):
            x[i].append(b[0][i])
        for i in range(len(y)):
            y[i].append(b[1][i])
    return (tuple(torch.from_numpy(np.stack(x[i])).float() for i in range(len(x))),
            tuple(torch.from_numpy(np.stack(y[i])).long() for i in range(len(y))))


def train_idm():
    assert torch.cuda.is_available()
    output_names = ["action", "on_ground", "has_jump", "has_flip"]
    train_dataset = IDMDataset(r"D:\rokutleg\idm-dataset", "train", mutate_action=True)
    val_dataset = IDMDataset(r"D:\rokutleg\idm-dataset", "validation", limit=1, shuffle=False, corrupt=False)

    ff_dim = 2048
    hidden_layers = 6
    dropout_rate = 0.
    lr = 5e-5
    batch_size = 300

    model = IDMNet(ff_dim, hidden_layers, dropout_rate)
    # lr *= np.sqrt(sum(p.numel() for p in model.parameters())) / 5
    print(model)
    model.cuda()
    # res = model.cuda()(torch.zeros(10, 77, 45).cuda(), torch.zeros(10, 9, 8).cuda())
    # model.cpu()
    # model = torch.jit.trace(model, (torch.zeros(10, 77, 45), torch.zeros(10, 9, 8))).cuda()
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss(reduce=False)
    scheduler = None  # LambdaLR(optimizer, lr_lambda=lambda e: 1 / (0.2 * e + 1))

    logger = wandb.init(group="idm", project="replay-model", entity="rolv-arild",
                        config=dict(ff_dim=ff_dim, hidden_layers=hidden_layers,
                                    dropout_rate=dropout_rate, lr=lr, batch_size=batch_size,
                                    optimizer=type(optimizer).__name__, lr_schedule=scheduler is not None))

    min_loss = float("inf")

    steps_per_hour = 108000
    assert steps_per_hour % batch_size == 0
    train_log_rate = 1
    val_log_rate = 24

    n = 0

    train_losses = {k: 0 for k in output_names}
    train_inferences = 0
    train_correct = {k: 0 for k in output_names}
    train_samples = 0
    l2_norm = 0
    model_mag = 0
    update_size = 0
    for epoch in range(100):
        model.train()
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate, drop_last=True)
        for x_train, y_train in train_loader:
            assert len(x_train[0]) == len(y_train[0]) == batch_size, f"{len(x_train)}, {len(y_train)}"
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
                        y_hat = model(x_val[0].cuda(), x_val[1].cuda())
                        for i, name in enumerate(output_names):
                            val_losses[name] += loss_fn(y_hat[i], y_val[i].cuda()).sum().item()
                            val_correct[name] += (y_hat[i].cpu().argmax(axis=-1) == y_val[i]).sum().item()
                        val_samples += len(x_val[0])
                        val_inferences += 1
                    val_losses = {k: v / val_samples for k, v in val_losses.items()}
                    accuracies = {k: v / val_samples for k, v in val_correct.items()}
                    loss = sum(val_losses.values())
                    logger.log({"validation/loss": loss}, commit=False)
                    logger.log({f"validation/{k}_loss": v for k, v in val_losses.items()}, commit=False)
                    logger.log({f"validation/{k}_accuracy": v for k, v in accuracies.items()}, commit=False)
                    print(f"Day {total_samples // (val_log_rate * steps_per_hour)}:", loss, val_losses)
                    if loss < min_loss:
                        mdl_jit = torch.jit.trace(model.cpu(), (torch.zeros(10, 77, 16), torch.zeros(10, 9, 8)))
                        torch.jit.save(mdl_jit, f"models/idm-model-{logger.name}.pt")
                        model.cuda()
                        print(f"Model saved at day {total_samples // (val_log_rate * steps_per_hour)} "
                              f"with total validation loss {loss}")
                        min_loss = loss
                model.train()

            y_hat = model(x_train[0].cuda(), x_train[1].cuda())
            loss = 0
            for i, name in enumerate(output_names):
                ls = loss_fn(y_hat[i], y_train[i].cuda())
                l = ls.mean()
                loss += l
                train_losses[name] += l.item()
                train_correct[name] += (y_hat[i].cpu().argmax(axis=-1) == y_train[i]).sum().item()
            train_inferences += 1
            train_samples += len(x_train[0])

            loss.backward()
            with torch.no_grad():
                gradient = [param.grad.clone() for param in model.parameters() if param.grad is not None]
                l2_norm += torch.sqrt(sum([(g ** 2).sum() for g in gradient])).item()
            # Store weights of model, so we can calculate the size of the update
            old_weights = [param.data.clone() for param in model.parameters()]

            optimizer.step()

            # Calculate the size of the update
            update_size += torch.sqrt(
                sum([((param.data - old) ** 2).sum() for param, old in zip(model.parameters(), old_weights)])).item()
            model_mag += torch.sqrt(sum([(param.data ** 2).sum() for param in model.parameters()])).item()

            model.zero_grad(True)
            n += 1

            if total_samples % (train_log_rate * steps_per_hour) == 0:
                train_losses = {k: v / train_inferences for k, v in train_losses.items()}
                # if train_losses["action"] > 1:
                #     print("Hei")
                #     print(y_hat[0].min().item())

                logger.log({"epoch": epoch}, commit=False)
                logger.log({"train/samples": total_samples}, commit=False)
                logger.log({"train/loss": sum(train_losses.values())}, commit=False)
                logger.log({f"train/{k}_loss": v for k, v in train_losses.items()}, commit=False)
                logger.log({f"train/{k}_accuracy": v / train_samples for k, v in train_correct.items()}, commit=False)
                logger.log({"train/update_size": update_size / train_inferences}, commit=False)
                logger.log({"train/l2_norm": l2_norm / train_inferences}, commit=False)
                logger.log({"train/model_mag": model_mag / train_inferences}, commit=False)
                logger.log({"train/learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else lr})
                print(f"Hour {total_samples // steps_per_hour}:", sum(train_losses.values()))
                train_losses = {k: 0 for k in output_names}
                train_inferences = 0
                train_correct = {k: 0 for k in output_names}
                train_samples = 0
                l2_norm = 0
                model_mag = 0
                update_size = 0
                if scheduler is not None:
                    scheduler.step()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    seed_everything(123)

    # make_idm_dataset(r"E:\rokutleg\idm-states-actions", r"D:\rokutleg\idm-dataset")
    train_idm()
    # test_idm()
