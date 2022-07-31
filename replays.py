import glob
import json
import os
import pickle
import re
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from typing import Iterator, Tuple

import numpy as np
import torch
from rlgym.utils.common_values import BOOST_LOCATIONS, BALL_RADIUS
from rlgym.utils.gamestates import GameState
from rlgym.utils.obs_builders import AdvancedObs

from main import get_data, rolling_window, make_lookup_table, normalize_quadrant, mirror_map, LIN_NORM, ANG_NORM

Replay = namedtuple("Replay", "metadata analyzer ball game players")


def quats_to_rot_mtx(quats: np.ndarray) -> np.ndarray:
    # From rlgym.utils.math.quat_to_rot_mtx
    w = -quats[:, 0]
    x = -quats[:, 1]
    y = -quats[:, 2]
    z = -quats[:, 3]

    theta = np.zeros((quats.shape[0], 3, 3))

    norm = np.einsum("fq,fq->f", quats, quats)

    sel = norm != 0

    w = w[sel]
    x = x[sel]
    y = y[sel]
    z = z[sel]

    s = 1.0 / norm[sel]

    # front direction
    theta[sel, 0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
    theta[sel, 1, 0] = 2.0 * s * (x * y + z * w)
    theta[sel, 2, 0] = 2.0 * s * (x * z - y * w)

    # left direction
    theta[sel, 0, 1] = 2.0 * s * (x * y - z * w)
    theta[sel, 1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
    theta[sel, 2, 1] = 2.0 * s * (y * z + x * w)

    # up direction
    theta[sel, 0, 2] = 2.0 * s * (x * z + y * w)
    theta[sel, 1, 2] = 2.0 * s * (y * z - x * w)
    theta[sel, 2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

    return theta


def get_data_df(df: pd.DataFrame, actions: np.ndarray):
    uids = [col.split("/")[0]
            for col in df.columns
            if not (col.startswith("ball") or col.startswith("invert")) and "pos_x" in col]
    positions = np.stack([df[[f"{uid}/pos_{axis}" for axis in "xyz"]].values for uid in uids]).swapaxes(0, 1)
    for i, uid in enumerate(uids):
        features = np.zeros((len(df), 45), dtype=float)
        labels = tuple(np.zeros(len(df), dtype=int) for _ in range(4))

        car_pos = df[[f"{uid}/pos_{axis}" for axis in "xyz"]].values
        features[:, 0:3] = car_pos * LIN_NORM
        features[:, 3:6] = df[[f"{uid}/vel_{axis}" for axis in "xyz"]].values * LIN_NORM
        rot = quats_to_rot_mtx(df[[f"{uid}/quat_{axis}" for axis in "wxyz"]].values)
        features[:, 6:9] = rot[:, :, 0]  # Forward
        features[:, 9:12] = rot[:, :, 2]  # Up
        features[:, 12:15] = df[[f"{uid}/ang_vel_{axis}" for axis in "xyz"]].values * ANG_NORM
        features[:, 15] = df[f"{uid}/boost_amount"].values
        features[:, 16] = df[f"{uid}/is_demoed"].values

        ball_pos = df[[f"ball/pos_{axis}" for axis in "xyz"]].values
        ball_close_mask = np.linalg.norm(ball_pos - car_pos, axis=-1) < 4 * BALL_RADIUS
        b_df = df[ball_close_mask]
        features[ball_close_mask, 17] = 1
        features[ball_close_mask, 18:21] = ball_pos[ball_close_mask] * LIN_NORM
        features[ball_close_mask, 21:24] = b_df[[f"ball/vel_{axis}" for axis in "xyz"]].values * LIN_NORM
        features[ball_close_mask, 24:27] = b_df[[f"ball/ang_vel_{axis}" for axis in "xyz"]].values * ANG_NORM

        dists = np.linalg.norm(positions - np.expand_dims(car_pos, 1), axis=-1)
        closest = np.argsort(dists)[:, 1]
        player_close_mask = dists[np.arange(len(dists)), closest] < 3 * BALL_RADIUS
        for p, puid in enumerate(uids):
            if puid == uid:
                continue
            p_mask = (closest == p) & player_close_mask
            p_df = df[p_mask]
            features[p_mask, 27] = 1
            features[p_mask, 28] = (p_df[f"{uid}/team_num"] == p_df[f"{puid}/team_num"]).values
            features[p_mask, 29:32] = p_df[[f"{puid}/pos_{axis}" for axis in "xyz"]].values * LIN_NORM
            features[p_mask, 32:35] = p_df[[f"{puid}/vel_{axis}" for axis in "xyz"]].values * LIN_NORM
            rot = quats_to_rot_mtx(p_df[[f"{puid}/quat_{axis}" for axis in "wxyz"]].values)
            features[p_mask, 35:38] = rot[:, :, 0]  # Forward
            features[p_mask, 38:41] = rot[:, :, 2]  # Up
            features[p_mask, 41:44] = p_df[[f"{puid}/ang_vel_{axis}" for axis in "xyz"]].values * ANG_NORM
            features[p_mask, 44] = p_df[f"{puid}/boost_amount"].values

        labels[0][:] = actions[:, i]
        labels[1][:] = df[f"{uid}/on_ground"].values
        labels[2][:] = df[f"{uid}/has_jump"].values
        labels[3][:] = df[f"{uid}/has_flip"].values
        yield features, labels


def load_parsed_replay(replay_folder):
    metadata = json.load(open(os.path.join(replay_folder, "metadata.json")))
    analyzer = json.load(open(os.path.join(replay_folder, "analyzer.json")))
    ball = pd.read_parquet(os.path.join(replay_folder, "__ball.parquet"))
    game = pd.read_parquet(os.path.join(replay_folder, "__game.parquet"))
    players = {}
    for player in metadata["players"]:
        uid = player["unique_id"]
        player_path = os.path.join(replay_folder, f"player_{uid}.parquet")
        if os.path.exists(player_path):
            players[uid] = pd.read_parquet(player_path)

    # Make and return named tuple
    return Replay(
        metadata, analyzer, ball, game, players
    )


def to_rlgym_dfs(parsed_replay: Replay):
    ball = parsed_replay.ball
    game = parsed_replay.game
    players = {}
    for uid, player_df in parsed_replay.players.items():
        player_df = player_df.copy()
        players[uid] = player_df

    df = pd.DataFrame(index=game.index)
    df.loc[:, "ticks_since_last_transmit"] = (game["delta"].values * 120).round()
    df.loc[:, ["blue_score", "orange_score"]] = 0
    df.loc[:, [f"pad_{n}" for n in range(34)]] = 0

    physics_cols = ["pos_x", "pos_y", "pos_z",
                    "quat_w", "quat_x", "quat_y", "quat_z",
                    "vel_x", "vel_y", "vel_z",
                    "ang_vel_x", "ang_vel_y", "ang_vel_z"]
    invert = np.array([-1, -1, 1,
                       -1, -1, 1,
                       -1, -1, 1])
    ball_physics_cols = [col for col in physics_cols if not col.startswith("quat")]

    ball_data = ball[ball_physics_cols].fillna(0).values
    df.loc[:, [f"ball/{col}" for col in ball_physics_cols]] = ball_data
    df.loc[:, [f"inverted_ball/{col}" for col in ball_physics_cols]] = ball_data * invert

    boost_locations = np.array(BOOST_LOCATIONS)

    controls_df = pd.DataFrame(index=df.index)
    player_metas = sorted(parsed_replay.metadata["players"], key=lambda x: int(x["unique_id"]))
    for player in player_metas:
        uid = player["unique_id"]
        player_df = players[uid]
        df.loc[:, f"{uid}/car_id"] = int(uid)
        df.loc[:, f"{uid}/team_num"] = int(player["is_orange"])
        df.loc[:, [f"{uid}/{col}" for col in physics_cols]] = player_df[physics_cols].values
        df.loc[:, [f"inverted_{uid}/{col}" for col in physics_cols]] = 0  # Get columns in right order first
        df.loc[:, [f"inverted_{uid}/{col}" for col in ball_physics_cols]] = (player_df[ball_physics_cols].values
                                                                             * invert)

        df.loc[:, [f"inverted_{uid}/quat_{axis}" for axis in "wxyz"]] = (
                player_df[["quat_z", "quat_y", "quat_x", "quat_w"]].values
                * np.array([-1, -1, 1, 1]))

        df.loc[:, f"{uid}/match_goals"] = player_df["match_goals"].fillna(0)
        df.loc[:, f"{uid}/match_saves"] = player_df["match_saves"].fillna(0)
        df.loc[:, f"{uid}/match_shots"] = player_df["match_shots"].fillna(0)
        df.loc[:, f"{uid}/match_demos"] = 0
        df.loc[:, f"{uid}/match_pickups"] = 0
        df.loc[:, f"{uid}/is_demoed"] = player_df["is_sleeping"].isna().astype(float)
        df.loc[:, f"{uid}/on_ground"] = False
        df.loc[:, f"{uid}/ball_touched"] = False
        df.loc[:, f"{uid}/has_jump"] = False
        df.loc[:, f"{uid}/has_flip"] = False
        df.loc[:, f"{uid}/boost_amount"] = player_df["boost_amount"].fillna(0) / 100

        ffill_cols = [f"{invert}{uid}/{col}" for invert in ("", "inverted_") for col in physics_cols]
        df.loc[:, ffill_cols] = df.loc[:, ffill_cols].ffill()

        controls_df.loc[:, f"{uid}/throttle"] = player_df["throttle"] / 127.5 - 1
        controls_df.loc[:, f"{uid}/steer"] = player_df["steer"] / 127.5 - 1
        controls_df.loc[:, f"{uid}/yaw"] = 0
        controls_df.loc[:, f"{uid}/pitch"] = 0
        controls_df.loc[:, f"{uid}/roll"] = 0
        controls_df.loc[:, f"{uid}/jump"] = 0
        controls_df.loc[:, f"{uid}/boost"] = player_df["boost_is_active"]
        controls_df.loc[:, f"{uid}/handbrake"] = player_df["handbrake"]

        for frame, pos in player_df[player_df["boost_pickup"] > 0][["pos_x", "pos_y", "pos_z"]].iterrows():
            boost_id = np.linalg.norm(boost_locations - pos.values, axis=-1).argmin()
            time_inactive = 10 if boost_locations[boost_id][2] > 72 else 4
            diff_time = game["time"] - game.loc[frame, "time"]
            inactive_period = diff_time.between(0, time_inactive, inclusive="left")
            df.loc[inactive_period, f"pad_{boost_id}"] = time_inactive - diff_time[inactive_period]
            df.loc[frame:, f"{uid}/match_pickups"] += 1

        demoed_diff = df[f"{uid}/is_demoed"].diff()
        for demo_frame, row in df[demoed_diff > 0].iterrows():
            respawn_frame = (demoed_diff.loc[demo_frame:] < 0).idxmax()
            timer1 = (3 - game.loc[demo_frame + 1: respawn_frame, "delta"].cumsum()).clip(lower=0)
            timer2 = (game.loc[demo_frame + 1: respawn_frame, "delta"][::-1]).cumsum()[::-1].clip(upper=3)

            df.loc[demo_frame: respawn_frame - 1, f"{uid}/is_demoed"] = np.maximum(timer1.values, timer2.values)
            # TODO add up match demos for attacker

    for goal in parsed_replay.metadata["game"]["goals"]:
        if goal["is_orange"]:
            df.loc[goal["frame"]:, "orange_score"] += 1
        else:
            df.loc[goal["frame"]:, "blue_score"] += 1
        player_id = next(p["unique_id"] for p in parsed_replay.metadata["players"] if p["name"] == goal["player_name"])
        df.loc[goal["frame"]:, f"{player_id}/match_goals"] += 1

    # DEPRECATED
    # for demo in parsed_replay.metadata["demos"]:
    #     frame = demo["frame_number"]
    #     attacker = demo["attacker_unique_id"]
    #     victim = demo["victim_unique_id"]
    #     time_dead = 3
    #     diff_time = game["time"] - game.loc[frame, "time"]
    #     dead_period = diff_time.between(0, time_dead, inclusive="left")
    #     df.loc[frame:, f"{attacker}/match_demos"] += 1
    #     df.loc[dead_period, f"{victim}/is_demoed"] = time_dead - diff_time[dead_period]

    for hit in parsed_replay.analyzer["hits"]:
        frame = hit["frame_number"]
        player_id = hit["player_unique_id"]
        df.loc[frame, f"{player_id}/ball_touched"] = True

    for gameplay_period in parsed_replay.analyzer["gameplay_periods"]:
        start_frame = gameplay_period["start_frame"]  # - 1
        end_frame = gameplay_period["goal_frame"]
        end_frame = df.index[-1] if end_frame is None else end_frame

        yield (df.loc[start_frame:end_frame - 1],
               controls_df[start_frame:end_frame])  # Actions are taken at t+1


def to_rlgym(df, controls_df=None):
    states = df.astype(float).apply(lambda x: GameState(list(x)), axis=1)
    if controls_df is None:
        yield from states
    else:
        actions = controls_df.astype(float).fillna(0).apply(lambda x: np.reshape(x.values, (-1, 8)), axis=1)
        yield from zip(
            states,
            actions
        )


def label_replay(parsed_replay: Replay):
    with torch.no_grad():
        it = to_rlgym_dfs(parsed_replay)
        for df, controls_df in it:
            states = list(to_rlgym(df))
            n_players = len(states[0].players)
            actions = np.zeros((len(states), n_players))
            for i, (x, _) in enumerate(get_data_df(df, actions)):
                x_rolled = x[rolling_window(np.arange(len(x)), 41, True, True)]
                mirrored = normalize_quadrant(x_rolled, [np.zeros((len(x), n_players))])
                inp = torch.from_numpy(x_rolled).float().cuda()
                y_hat = [0.] * 4
                for _ in range(40):
                    t = model(inp)
                    for j in range(4):
                        y_hat[j] += t[j]
                pred_actions, pred_on_ground, pred_has_jump, pred_has_flip = (t.argmax(axis=-1).cpu().numpy()
                                                                              for t in y_hat)
                pred_actions[mirrored] = mirror_map[pred_actions[mirrored]]
                for j, state in enumerate(states):
                    state.players[i].on_ground = pred_on_ground[j] == 1
                    state.players[i].has_jump = pred_has_jump[j] == 1
                    state.players[i].has_flip = pred_has_flip[j] == 1
                actions[:, i] = pred_actions
            yield zip(states, actions)


def make_dataset(input_folder, output_folder, shard_size=30 * 60 * 60):
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


def manual_validate():
    replay_path = r"D:\rokutleg\parsed\2021-electrum-replays\ranked-doubles\gold\00a43335-dfb0-49ca-ab48-aa2f16a38d9c.replay\00a43335-dfb0-49ca-ab48-aa2f16a38d9c"
    parsed_replay = load_parsed_replay(replay_path)
    list(label_replay(parsed_replay))
    it = to_rlgym_dfs(parsed_replay)
    data = []
    dfs = []
    with torch.no_grad():
        # model.eval()
        for df, segment in it:
            states = list(to_rlgym(df))
            n_players = len(states[0].players)
            actions = np.zeros((len(states), n_players))
            for i, (x, _) in enumerate(get_data(states, actions)):
                x_rolled = x[rolling_window(np.arange(len(x)), 41, True, True)]
                mirrored = normalize_quadrant(x_rolled, [np.zeros((len(x), n_players))])
                inp = torch.from_numpy(x_rolled).float().cuda()
                y_hat = [0.] * 4
                for _ in range(40):
                    t = model(inp)
                    for j in range(4):
                        y_hat[j] += t[j]
                pred_actions, pred_on_ground, pred_has_jump, pred_has_flip = (t.argmax(axis=-1).cpu().numpy()
                                                                              for t in y_hat)
                pred_actions[mirrored] = mirror_map[pred_actions[mirrored]]
                actions[:, i] = pred_actions
            data.append((states, actions))
            dfs.append(df)
    players = sorted(parsed_replay.metadata["players"], key=lambda x: int(x["unique_id"]))
    df = pd.concat(dfs)
    action_indices = np.concatenate([a for s, a in data]).astype(int)
    actions = lut[action_indices, :]
    df[[f"{player['unique_id']}/action_index" for player in players]] = action_indices
    buttons = "THROTTLE, STEER, PITCH, YAW, ROLL, JUMP, BOOSTING, HANDBRAKE".lower().split(", ")
    df[[f"{player['unique_id']}/action_{button}" for player in players for button in
        buttons]] = actions.reshape(actions.shape[0], -1)
    df[[f"{player['unique_id']}/name" for player in players]] = [player["name"] for player in players]
    df.to_parquet("states_actions_005e01cc-6a84-43d7-81c8-2bd29ac4d392.parquet")


if __name__ == '__main__':
    lut = make_lookup_table()
    model = torch.jit.load("idm-model.pt").cuda()
    # manual_validate()

    make_dataset(r"D:\rokutleg\parsed\2021-electrum-replays\ranked-doubles",
                 r"D:\rokutleg\electrum-dataset")
