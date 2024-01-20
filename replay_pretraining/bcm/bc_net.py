from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


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
