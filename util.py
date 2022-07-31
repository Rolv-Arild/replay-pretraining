from typing import List

import numpy as np
from rlgym.utils.common_values import BALL_RADIUS
from rlgym.utils.gamestates.game_state import GameState

from behavioral_cloning import train_bc


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
