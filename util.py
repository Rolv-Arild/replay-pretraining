import re
from collections import namedtuple
from typing import List

import numpy as np
import torch.jit
from rlgym.utils.common_values import BALL_RADIUS, BLUE_TEAM
from rlgym.utils.gamestates.game_state import GameState


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

mirror_map = np.array([np.where((lookup_table == action).all(axis=-1))[0][0]
                       for action in lookup_table * np.array([1, -1, 1, -1, -1, 1, 1, 1])], dtype=int)


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


def encoded_states_to_advanced_obs(df, actions):
    uids = [col.split("/")[0]
            for col in df.columns
            if not (col.startswith("ball") or col.startswith("invert")) and "pos_x" in col]
    obs = np.zeros((len(uids), len(df), 51 + 25 + 31 * (len(uids) - 1)))

    pos_std = 2300
    ang_std = np.pi

    for i, uid in enumerate(uids):
        def add_player(player_id, idx):
            pos = df[[f"{invert}{player_id}/pos_{axis}" for axis in "xyz"]].values
            vel = df[[f"{invert}{player_id}/vel_{axis}" for axis in "xyz"]].values
            obs[i, :, idx:idx + 3] = (ball_pos - pos) / pos_std
            obs[i, :, idx + 3:idx + 6] = (ball_vel - vel) / pos_std
            obs[i, :, idx + 6:idx + 9] = pos / pos_std
            rot = quats_to_rot_mtx(df[[f"{player_id}/quat_{axis}" for axis in "wxyz"]].values)
            obs[i, :, idx + 9:idx + 12] = rot[:, :, 0]  # Forward
            obs[i, :, idx + 12:idx + 15] = rot[:, :, 2]  # Up
            obs[i, :, idx + 15:idx + 18] = vel / pos_std
            obs[i, :, idx + 18:idx + 21] = (df[[f"{invert}{player_id}/ang_vel_{axis}" for axis in "xyz"]].values
                                            / ang_std)
            obs[i, :, idx + 21] = df[f"{player_id}/boost_amount"]
            obs[i, :, idx + 22] = df[f"{player_id}/on_ground"]
            obs[i, :, idx + 23] = df[f"{player_id}/has_flip"]
            obs[i, :, idx + 24] = df[f"{player_id}/is_demoed"]

        invert = "" if df[f"{uid}/team_num"].iloc[0] == BLUE_TEAM else "inverted_"

        ball_pos = df[[f"{invert}ball/pos_{axis}" for axis in "xyz"]].values
        ball_vel = df[[f"{invert}ball/vel_{axis}" for axis in "xyz"]].values
        obs[i, :, 0:3] = ball_pos / pos_std
        obs[i, :, 3:6] = ball_vel / pos_std
        obs[i, :, 6:9] = df[[f"{invert}ball/ang_vel_{axis}" for axis in "xyz"]].values / ang_std
        obs[i, :, 9:17] = lookup_table[[8] + actions[:-1, i].astype(int).tolist()]
        obs[i, :, 17:51] = df[[f"pad_{33 - n if invert else n}" for n in range(34)]].values / 10

        index = 51
        add_player(uid, index)
        index += 25

        teams = {puid: df[f"{puid}/team_num"].iloc[0] == df[f"{uid}/team_num"].iloc[0] for puid in uids}
        for puid in sorted(uids, key=teams.get):
            if puid == uid:
                continue
            add_player(puid, index)
            index += 25
            obs[i, :, index:index + 3] = (df[[f"{invert}{puid}/pos_{axis}" for axis in "xyz"]].values
                                          - df[[f"{invert}{uid}/pos_{axis}" for axis in "xyz"]].values) / pos_std
            obs[i, :, index + 3:index + 6] = (df[[f"{invert}{puid}/vel_{axis}" for axis in "xyz"]].values
                                              - df[[f"{invert}{uid}/vel_{axis}" for axis in "xyz"]].values) / pos_std
            index += 6

    yield from ((obs[i], actions[:, i]) for i in range(len(uids)))


idm_model = torch.jit.load("idm-model-super-star-16.pt").to("cuda" if torch.cuda.is_available() else "cpu")
Replay = namedtuple("Replay", "metadata analyzer ball game players")
