import re
from collections import namedtuple
from typing import List, Optional, Any

import numpy as np
import torch.jit
from numba import njit
from rlgym_sim.utils.common_values import BALL_RADIUS, BLUE_TEAM, CEILING_Z, BACK_WALL_Y
from rlgym_sim.utils.gamestates.game_state import GameState
from torch import nn

INVERT_SIDE_ACTIONS = np.array([1, -1, 1, -1, -1, 1, 1, 1])
BUTTONS = ['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake']


def make_lookup_table(throttle_bins: Any = 3,
                      steer_bins: Any = 3,
                      torque_bins: Any = 3,
                      flip_bins: Any = 8,
                      include_stall=False):
    # Parse bins
    def parse_bin(b, endpoint=True):
        if isinstance(b, int):
            b = np.linspace(-1, 1, b, endpoint=endpoint)
        else:
            b = np.array(b)
        return b

    throttle_bins = parse_bin(throttle_bins)
    steer_bins = parse_bin(steer_bins)
    torque_bins = parse_bin(torque_bins)
    flip_bins = (parse_bin(flip_bins, endpoint=False) + 1) * np.pi  # Split a circle into equal segments in [0, 2pi)

    actions = []

    # Ground
    pitch = roll = jump = 0
    for throttle in throttle_bins:
        for steer in steer_bins:
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    yaw = steer
                    actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])

    # Aerial
    jump = handbrake = 0
    for pitch in torque_bins:
        for yaw in torque_bins:
            for roll in torque_bins:
                if pitch == roll == 0 and np.isclose(yaw, steer_bins).any():
                    continue  # Duplicate with ground
                magnitude = max(abs(pitch), abs(yaw), abs(roll))
                if magnitude < 1:
                    continue  # Duplicate rotation direction, only keep max magnitude
                for boost in (0, 1):
                    throttle = boost
                    steer = yaw
                    actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])

    # Flips and jumps
    jump = handbrake = 1  # Enable handbrake for potential wavedashes
    yaw = steer = 0  # Only need roll for sideflip
    angles = [np.nan] + [v for v in flip_bins]
    for angle in angles:
        if np.isnan(angle):
            pitch = roll = 0  # Empty jump
        else:
            pitch = np.sin(angle)
            roll = np.cos(angle)
            # Project to square of diameter 2 because why not
            magnitude = max(abs(pitch), abs(roll))
            pitch /= magnitude
            roll /= magnitude
        for boost in (0, 1):
            throttle = boost
            actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])
    if include_stall:
        actions.append([0, 0, 0, 1, -1, 1, 0, 1])  # One final action for stalling

    actions = np.round(actions, 3)  # Convert to numpy and remove floating point errors
    assert len(np.unique(actions, axis=0)) == len(actions), 'Duplicate actions found'

    return actions


LIN_NORM = 1 / 2300
ANG_NORM = 1 / 5.5
lookup_table = make_lookup_table()

mirror_map = np.array([np.where((lookup_table == action).all(axis=-1))[0][0]
                       for action in lookup_table * np.array([1, -1, 1, -1, -1, 1, 1, 1])], dtype=int)


def dist_to_walls(x, y):
    dist_side_wall = abs(4096 - abs(x))
    dist_back_wall = abs(5120 - abs(y))

    # From https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    x1, y1, x2, y2 = 4096 - 1152, 5120, 4096, 5120 - 1152  # Line segment for corner
    A = abs(x) - x1
    B = abs(y) - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:  # in case of 0 length line
        param = dot / len_sq

    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = abs(x) - xx
    dy = abs(y) - yy
    dist_corner_wall = np.sqrt(dx * dx + dy * dy)

    return dist_side_wall, dist_back_wall, dist_corner_wall


def group_field_locations(features, label):
    # TODO finish?
    calculate_wall_dists = np.vectorize(dist_to_walls)

    mid = features.shape[1] // 2
    positions = features[:, mid, 0:3]
    ball_close = features[:, mid, 17]

    threshold = 100 + ball_close * 2 * BALL_RADIUS
    dist_side_wall, dist_back_wall, dist_corner_wall = calculate_wall_dists(positions[..., 0], positions[..., 1])
    dist_floor = positions[..., 2]
    dist_ceiling = CEILING_Z - positions[..., 2]

    outside_goal = positions[..., 1] < BACK_WALL_Y

    far_from_corner = dist_corner_wall > threshold
    far_from_side_wall = dist_side_wall > threshold
    far_from_back_wall = dist_back_wall > threshold
    features[outside_goal & far_from_corner & far_from_side_wall, 0] = 0
    features[outside_goal & far_from_corner & far_from_back_wall, 1] = 0

    far_from_floor = dist_floor > threshold
    far_from_ceiling = dist_ceiling > threshold
    features[outside_goal & far_from_floor & far_from_ceiling, 2] = CEILING_Z / 2


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

    actions[mirrored] *= INVERT_SIDE_ACTIONS

    return mirrored


# v is the magnitude of the velocity in the car's forward direction
def curvature(v):
    if 0.0 <= v < 500.0:
        return 0.006900 - 5.84e-6 * v
    if 500.0 <= v < 1000.0:
        return 0.005610 - 3.26e-6 * v
    if 1000.0 <= v < 1500.0:
        return 0.004300 - 1.95e-6 * v
    if 1500.0 <= v < 1750.0:
        return 0.003025 - 1.1e-6 * v
    if 1750.0 <= v < 2500.0:
        return 0.001800 - 4e-7 * v

    return 0.0


def equivalent(player, true_action, alt_action, threshold=0.):
    if player.is_demoed:
        return True

    assert 0 <= threshold <= 1

    true_linear_acceleration = np.zeros(3)
    true_angular_acceleration = np.zeros(3)
    alt_linear_acceleration = np.zeros(3)
    alt_angular_acceleration = np.zeros(3)

    for action, lin_acc, ang_acc in ((true_action, true_linear_acceleration, true_angular_acceleration),
                                     (alt_action, alt_linear_acceleration, alt_angular_acceleration)):
        throttle, steer, pitch, yaw, roll, jump, boost, handbrake = action

        # Throttle
        if boost > 0 and player.boost_amount > 0:
            throttle = 1
        forward_speed = player.car_data.linear_velocity @ player.car_data.forward()
        if player.on_ground:
            if forward_speed * throttle < 0:
                acc = -3500  # Braking, e.g. accelerating in the opposite direction
            elif abs(throttle) < 0.01 and forward_speed != 0:
                acc = -525  # Coasting deceleration
            else:
                abs_fs = abs(forward_speed)
                if abs_fs <= 1400:
                    acc = 1600 - abs_fs * (1600 - 160) / 1400
                elif abs_fs <= 1410:
                    acc = 160 - (abs_fs - 1400) * (160 - 0) / 10
                else:
                    acc = 0
                acc *= throttle
        else:
            acc = 66.667 * throttle if throttle >= 0 else 33.334 * throttle
        lin_acc += acc * player.car_data.forward()

        # Steer
        if player.on_ground:
            is_still = (true_action[0] == 0) and (player.car_data.linear_velocity == 0).all()
            if not is_still:
                forward_speed += lin_acc @ player.car_data.forward() / 2
                c = curvature(abs(forward_speed))
                turn_radius = 1 / c

        # Pitch, yaw, roll
        if not player.on_ground:
            if (player.has_flip or player.has_jump
                    and true_action[5] == 1 and abs(true_action[2:5]).sum() > 0.5
                    and alt_action[5] == 1 and abs(alt_action[2:5]).sum() > 0.5):
                dx = dy = 0
                dz = -player.car_data.linear_velocity[2]  # Cancel vertical velocity
                if true_action[3] == -true_action[4] and true_action[2] == 0:
                    pass  # Stall
                else:
                    # Only direction matters
                    flip_dir = true_action[2:5] / np.linalg.norm(true_action[2:5])
                    dx
                lin_acc += np.array([0, 0, ])  # Cancel vertical velocity
            else:
                pitch_acc = 12.46 * pitch
                yaw_acc = 9.11 * yaw
                roll_acc = 38.34 * roll

                ang_acc += np.array([pitch_acc, yaw_acc, roll_acc])

        # Jump
        if player.on_ground or player.has_flip or player.has_jump:
            total_diff += (true_action[5] - alt_action[5]) ** 2

        # Boost
        if player.boost_amount > 0:
            total_diff += (true_action[6] - alt_action[6]) ** 2

        # Handbrake
        if player.on_ground:
            is_still = (true_action[0] == alt_action[0] == 0) and (player.car_data.linear_velocity == 0).all()
            if not is_still:
                total_diff += (true_action[7] - alt_action[7]) ** 2

    return np.sqrt(total_diff) <= threshold


def get_data(states: List["GameState"], actions: np.ndarray, action_options: int = 16):
    positions = np.array([[p.car_data.position for p in state.players] for state in states])
    for i in range(len(states[0].players)):
        x_data = np.zeros((len(states), 45))
        y_data = ((np.zeros((len(states), 8)),
                   np.zeros((len(states), action_options, 8)),
                   np.zeros((len(states), action_options, 8))) +
                  tuple(np.zeros(len(states)) for _ in range(3)))
        j = 0
        for state in states:
            player = state.players[i]
            if player.is_demoed:
                continue
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

            random_actions = np.zeros((action_options, 8))
            for k in range(action_options):
                while True:
                    act = random_action()
                    if not equivalent(player, action, act, 0.25):
                        break
                random_actions[k] = act
            mutated_actions = np.zeros((action_options, 8))
            for k in range(action_options):
                while True:
                    act = mutate_action(action)
                    if not equivalent(player, action, act, 0.25):
                        break
                mutated_actions[k] = act
            random_actions_noneq = np.zeros((action_options, 8))
            for k in range(action_options):
                while True:
                    act = random_action()
                    if not equivalent(player, action, act, 0.25):
                        break
                random_actions_noneq[k] = act
            mutated_actions_noneq = np.zeros((action_options, 8))
            for k in range(action_options):
                while True:
                    act = mutate_action(action)
                    if not equivalent(player, action, act, 0.25):
                        break
                mutated_actions_noneq[k] = act

            x_data[j] = features
            y_data[0][j] = action
            y_data[1][j] = random_actions
            y_data[2][j] = mutated_actions
            y_data[3][j] = random_actions_noneq
            y_data[4][j] = mutated_actions_noneq
            y_data[5][j] = player.on_ground
            y_data[6][j] = player.has_jump
            y_data[7][j] = player.has_flip
            j += 1

        # Cut off invalid data
        x_data = x_data[:j]
        y_data = tuple(y[:j] for y in y_data)
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
    # from rlgym_sim.utils.math.quat_to_rot_mtx
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


class ControlsPredictorDot(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=32, layers=1, actions=None):
        super().__init__()
        if actions is not None:
            self.actions = torch.from_numpy(actions).float()
        else:
            self.actions = actions
        self.invert_side_actions = torch.from_numpy(INVERT_SIDE_ACTIONS).float()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_features),
            *sum(([nn.GELU(), nn.Linear(hidden_features, hidden_features)] for _ in range(layers)), []),
            nn.GELU(),
            nn.Linear(hidden_features, out_features)
        )
        self.emb_convertor = nn.Linear(in_features, out_features)

    def forward(self, player_emb: torch.Tensor, actions: Optional[torch.Tensor] = None, flip=None):
        if actions is None:
            if self.actions is not None:
                actions = self.actions
            else:
                raise ValueError("Need to supply action options")
        actions = actions.to(player_emb.device)
        if flip is not None:
            if actions.ndim == 2:
                actions = actions.unsqueeze(0).repeat(player_emb.shape[0], 1, 1)
            actions[flip] *= self.invert_side_actions.to(player_emb.device)
        player_emb = self.emb_convertor(player_emb)
        act_emb = self.net(actions)

        if act_emb.ndim == 2:
            return torch.einsum("ad,bd->ba", act_emb, player_emb)

        return torch.einsum("bad,bd->ba", act_emb, player_emb)


class ControlsPredictorLinear(nn.Module):
    def __init__(self, in_features, features=32, layers=1, actions=None):
        super().__init__()
        if actions is not None:
            self.actions = torch.from_numpy(actions).float()
        else:
            self.actions = actions
        self.invert_side_actions = torch.from_numpy(INVERT_SIDE_ACTIONS).float()
        self.net = nn.Sequential(
            nn.Linear(in_features + 8, features),
            *sum(([nn.ReLU(), nn.Linear(features, features)] for _ in range(layers)), []),
            nn.ReLU(),
            nn.Linear(features, 1)
        )

    def forward(self, player_emb: torch.Tensor, actions: Optional[torch.Tensor] = None, flip=None):
        if actions is None:
            if self.actions is not None:
                actions = self.actions
            else:
                raise ValueError("Need to supply action options")
        actions = actions.to(player_emb.device)
        if flip is not None:
            if actions.ndim == 2:
                actions = actions.unsqueeze(0).repeat(player_emb.shape[0], 1, 1)
            actions[flip] *= self.invert_side_actions.to(player_emb.device)
        player_emb = player_emb.unsqueeze(1).repeat(1, actions.shape[1], 1)

        x = torch.cat((player_emb, actions), axis=2)

        y = self.net(x)
        return y.squeeze(2)


# idm_model = torch.jit.load("models/idm-model-icy-paper-137.pt").to("cuda" if torch.cuda.is_available() else "cpu")
Replay = namedtuple("Replay", "metadata analyzer ball game players")


def random_action():
    a = np.zeros(8)
    a = mutate_action(a, np.arange(8).astype(int))
    return a


# @njit
def mutate_action(a, indices=None):
    a = np.copy(a)
    if indices is None:
        # indices = range(8)
        indices = np.where(np.random.random(size=8) < 1 / 8)[0].astype(np.int32)
    for i in indices:
        if i < 5:
            r = np.random.random()
            cutoffs = [0.2 ** 2, 0.3 ** 2, 0.5 ** 2, 0.6 ** 2] if i == 0 else [0.2, 0.3, 0.7, 0.8]
            if r < cutoffs[0]:
                a[i] = -1
            elif r < cutoffs[1]:
                a[i] = -np.random.random()
            elif r < cutoffs[2]:
                a[i] = 0
            elif r < cutoffs[3]:
                a[i] = +np.random.random()
            else:
                a[i] = +1
        elif i == 5:
            a[5] = np.random.random() < 0.2
        elif i == 6:
            a[6] = np.random.random() < 0.2
        elif i == 7:
            a[7] = np.random.random() < 0.2
    return a
