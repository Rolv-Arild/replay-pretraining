import argparse
import os
import random
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Iterator

import numba as numba
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co

from replay_pretraining.utils.game_state import GameState
from replay_pretraining.utils.util import get_data, rolling_window, normalize_quadrant


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


def process_episode(states, actions, window_size=38):
    states = [GameState(arr.tolist()) for arr in states]

    pad = False

    result = []
    for x, y in get_data(states, actions):
        if not pad and len(x) < 2 * window_size + 1:
            continue
        window = rolling_window(np.arange(len(x)), 2 * window_size + 1, pad_start=pad, pad_end=pad)
        grouped_x = x[window]
        grouped_y = tuple(y[i][window[:, window_size]] for i in range(len(y)))
        normalize_quadrant(grouped_x, grouped_y)
        assert grouped_y[0].max() <= 1

        result.append((grouped_x, grouped_y))
    return result


def process_file(path, window_size=38, workers=1):
    try:
        f = np.load(path)
        all_states = f["states"]
        all_episodes = f["episodes"]
        all_actions = f["actions"].reshape(-1, 6, 8)
        futures = []
        with ProcessPoolExecutor(workers) as ex:
            for episode in np.unique(all_episodes):
                mask = all_episodes == episode

                states = all_states[mask]
                actions = all_actions[mask]

                future = ex.submit(process_episode, states, actions, window_size)
                futures.append(future)
    except (zipfile.BadZipFile, EOFError):
        return []
    return [f.result() for f in futures]


def make_idm_dataset(in_folder, out_folder, shard_size=60 * 60 * 30, workers=1):
    train = [[], 0, "train"]
    validation = [[], 0, "validation"]
    test = [[], 0, "test"]
    splits = [train, validation, test]
    window_size = 38
    paths = [os.path.join(in_folder, file) for file in os.listdir(in_folder)]
    for path in paths:
        results = process_file(path, window_size, workers)
        for result in results:
            for x, y in result:
                split = splits[np.random.choice(len(splits), p=[0.98, 0.01, 0.01])]
                split[0].append((x, y))
                if sum(len(gx) for gx, gy in split[0]) > shard_size:
                    x_data = np.concatenate([gx for gx, gy in split[0]])
                    y_data = tuple(np.concatenate([gy[i] for gx, gy in split[0]])
                                   for i in range(len(split[0][0][1])))
                    out_name = f"{split[2]}-shard-{split[1]}.npz"
                    print(f"{out_name} ({len(x_data)})")
                    np.savez_compressed(os.path.join(out_folder, out_name), features=x_data,
                                        actions=y_data[0], random_actions=y_data[1], mutated_actions=y_data[2],
                                        on_ground=y_data[3], has_jump=y_data[4], has_flip=y_data[5])
                    split[1] += 1
                    split[0].clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", required=True)
    parser.add_argument("--out_folder", required=True)
    parser.add_argument("--shard_size", type=int, default=60 * 60 * 30)
    parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    make_idm_dataset(args.in_folder, args.out_folder, args.shard_size, args.workers)
