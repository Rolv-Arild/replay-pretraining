import os
import zipfile
from typing import Iterator

import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co


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
