import functools
import json
import os
import warnings

import pandas as pd

import numpy as np
from rlgym_sim.utils.common_values import BOOST_LOCATIONS, BALL_RADIUS
from rlgym_sim.utils.gamestates import GameState

from replay_pretraining.replays.inverse_aerial_controls import pyr_from_dataframe
from replay_pretraining.utils.util import LIN_NORM, ANG_NORM, \
    quats_to_rot_mtx, Replay


def get_data_df(df: pd.DataFrame, actions: np.ndarray):
    uids = sorted([col.split("/")[0]
                   for col in df.columns
                   if not (col.startswith("ball") or col.startswith("invert")) and "pos_x" in col])
    positions = np.stack([df[[f"{uid}/pos_{axis}" for axis in "xyz"]].values for uid in uids]).swapaxes(0, 1)
    for i, uid in enumerate(uids):
        features = np.zeros((len(df), 45), dtype=float)
        labels = (np.zeros((len(df), 8), dtype=int),) + tuple(np.zeros(len(df), dtype=int) for _ in range(3))

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

        closest = np.full(len(df), -1, dtype=int)
        player_close_mask = np.full(len(df), False, dtype=bool)
        if positions.shape[1] > 1:
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
        labels[1][:] = df[f"{uid}/on_ground"].fillna(False).values
        labels[2][:] = df[f"{uid}/has_jump"].fillna(False).values
        labels[3][:] = df[f"{uid}/has_flip"].fillna(False).values
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


def to_rlgym_dfs(parsed_replay: Replay, lookup_table=None):
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        ball = parsed_replay.ball
        game = parsed_replay.game
        players = parsed_replay.players

        df = pd.DataFrame(index=game.index)
        df.loc[:, "seconds_remaining"] = game["seconds_remaining"].astype(float)
        df.loc[game["is_overtime"].fillna(0.).astype(bool), "seconds_remaining"] = np.inf
        df.loc[:, ["blue_score", "orange_score"]] = 0
        df.loc[:, [f"pad_{n}" for n in range(34)]] = 0.

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

        controls = {}

        player_metas = sorted(parsed_replay.metadata["players"], key=lambda x: int(x["unique_id"]))
        for player in player_metas:
            uid = player["unique_id"]
            if uid not in players:
                continue
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
            df.loc[:, f"{uid}/on_ground"] = np.nan
            df.loc[:, f"{uid}/ball_touched"] = False
            df.loc[:, f"{uid}/has_jump"] = np.nan
            df.loc[:, f"{uid}/has_flip"] = np.nan
            df.loc[:, f"{uid}/boost_amount"] = player_df["boost_amount"].fillna(0) / 100

            ffill_cols = [f"{invert}{uid}/{col}" for invert in ("", "inverted_") for col in physics_cols]
            df.loc[:, ffill_cols] = df.loc[:, ffill_cols].ffill()

            jumping = player_df["jump_is_active"].fillna(0.) > 0
            dodging = player_df["dodge_is_active"].fillna(0.) > 0
            double_jumping = player_df["double_jump_is_active"].fillna(0.).diff() > 0  # DJ only acts for one frame
            flipping_car = player_df["flip_car_is_active"].fillna(0.) > 0

            df.loc[jumping.diff() > 0, [f"{uid}/on_ground", f"{uid}/has_jump", f"{uid}/has_flip"]] = 1
            df.loc[double_jumping | dodging | flipping_car, f"{uid}/on_ground"] = 0
            df.loc[dodging, f"{uid}/has_flip"] = 0
            df.loc[double_jumping | dodging.diff() > 0, f"{uid}/has_flip"] = 1

            controls[uid] = get_actions_from_player(player_df, game["delta"], lookup_table)

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
            player_id = next(
                p["unique_id"] for p in parsed_replay.metadata["players"] if p["name"] == goal["player_name"])
            df.loc[goal["frame"]:, f"{player_id}/match_goals"] += 1

        for hit in parsed_replay.analyzer["hits"]:
            frame = hit["frame_number"]
            player_id = hit["player_unique_id"]
            df.loc[frame, f"{player_id}/ball_touched"] = True

        for gameplay_period in parsed_replay.analyzer["gameplay_periods"]:
            start_frame = gameplay_period["start_frame"]  # - 1
            end_frame = gameplay_period["goal_frame"]
            end_frame = df.index[-1] if end_frame is None else end_frame

            if lookup_table is None:
                c = {
                    uid: controls[uid].loc[start_frame:end_frame - 1].copy()
                    for uid in controls
                }
            else:
                c = {
                    uid: (controls[uid][0].loc[start_frame:end_frame - 1].copy(),
                          controls[uid][1][:, start_frame:end_frame])
                    for uid in controls
                }

            gamestate_df = df.loc[start_frame:end_frame - 1].copy()
            yield gamestate_df, c


def get_actions_from_player(player_df, deltas, lookup_table=None):
    # For throttle and steer, it goes from 0 to 255, so we need to normalize it to -1 to 1. This should give
    # the closest approximation except for the case of 127/128, which typically means 0 (127.5 rounded)
    throttle = player_df["throttle"].fillna(127.5) / 127.5 - 1
    throttle *= abs(throttle) > 0.01
    steer = player_df["steer"].fillna(127.5) / 127.5 - 1
    steer *= abs(steer) > 0.01

    # All the kinds of jumps
    jumping = player_df["jump_is_active"].fillna(0.) > 0
    dodge_active = player_df["dodge_is_active"].fillna(0.)
    dodging = dodge_active.diff() > 0
    double_jumping = player_df["double_jump_is_active"].fillna(0.).diff() > 0
    flipping_car = player_df["flip_car_is_active"].fillna(0.) > 0
    jump_button_pressed = jumping | dodging | double_jumping | flipping_car

    # Dodges
    dodge_torques = player_df.loc[:, ["dodge_torque_x", "dodge_torque_y", "dodge_torque_z"]].values
    dodge_torques[~dodging] = 0
    dodge_direction = -dodge_torques
    mx = np.max(np.abs(dodge_direction), axis=1, keepdims=True)
    msk = mx[..., 0] != 0
    dodge_direction[msk] /= mx[msk]
    roll = dodge_direction[:, 0]
    pitch = dodge_direction[:, 1]
    yaw = np.zeros_like(roll)
    yaw[~msk & dodging] = 1  # This is just to handle stalls
    roll[~msk & dodging] = -1

    # Aerial controls
    pyr = pyr_from_dataframe(deltas, player_df)

    # Double jumps
    dodge_deadzone = 0.5  # Default dodge deadzone in RLGym/RLBot
    mag = np.abs(pyr).sum(axis=1)
    mask = double_jumping & (mag > dodge_deadzone - 0.01)
    pyr[mask, :] = (dodge_deadzone - 0.01) * pyr[mask, :] / mag[mask].reshape(-1, 1)

    boost = player_df["boost_is_active"].fillna(0.)
    handbrake = player_df["handbrake"].astype(float).fillna(0.)

    controls_df = pd.DataFrame(index=player_df.index)

    controls_df.loc[:, f"throttle"] = throttle
    controls_df.loc[:, f"steer"] = steer
    controls_df.loc[:, f"pitch"] = pyr[:, 0]
    controls_df.loc[dodging, f"pitch"] = pitch[dodging]
    controls_df.loc[:, f"yaw"] = pyr[:, 1]
    controls_df.loc[dodging, f"yaw"] = yaw[dodging]
    controls_df.loc[:, f"roll"] = pyr[:, 2]
    controls_df.loc[dodging, f"roll"] = roll[dodging]
    # controls_df.loc[double_jumping, [f"pitch", f"yaw", f"roll"]] = 0
    controls_df.loc[:, f"jump"] = jump_button_pressed.astype(float)
    controls_df.loc[:, f"boost"] = boost
    controls_df.loc[:, f"handbrake"] = handbrake

    if lookup_table is None:
        return controls_df

    assert player_df.index[0] == 0

    # If lookup table is provided, return plausibility ratings for each action for each frame instead of raw actions
    # It is split by ratings for jumps, grounded, and aerial so they can later be combined using the IDM predictions
    actions = np.tile(lookup_table, (player_df.shape[0], 1, 1))
    jm = jumping.values
    dm = dodging.values
    djm = double_jumping.values
    fcm = flipping_car.values

    ratings_jumps = np.zeros(actions.shape[:2])
    ratings_grounded = np.zeros(actions.shape[:2])
    ratings_aerial = np.zeros(actions.shape[:2])

    # Make sure it's not jumping when it shouldn't be
    pressing_jump = jm | dm | djm | fcm
    ratings_jumps[~pressing_jump] -= actions[~pressing_jump, :, 5]

    # Make sure it's jumping when it should be
    ratings_jumps[jm | fcm] += actions[jm | fcm, :, 5]

    # Make sure it's double jumping with pitch/yaw/roll below deadzone
    below_deadzone = np.abs(actions[djm, :, 2:5]).sum(axis=-1) < dodge_deadzone
    ratings_jumps[djm] += actions[djm, :, 5] * below_deadzone

    # Make sure it's dodging in the right direction
    above_deadzone = np.abs(actions[dm, :, 2:5]).sum(axis=-1) >= dodge_deadzone
    true_dir = dodge_direction[dodging][..., :2].reshape(-1, 1, 2)
    norm_true = np.linalg.norm(true_dir, axis=-1)
    pred_dir = np.concatenate([actions[dm, :, 3:4] + actions[dm, :, 4:5],
                               actions[dm, :, 2:3]], axis=-1)  # (yaw + roll, pitch)
    norm_pred = np.linalg.norm(actions[dm, :, 2:5], axis=-1)
    divisor = norm_true * norm_pred
    with np.errstate(divide='ignore', invalid='ignore'):
        cosine_sim = np.where(
            divisor != 0,
            (true_dir * pred_dir).sum(axis=-1) / divisor,  # Normal dodge
            np.all(true_dir == pred_dir, axis=-1)  # Stall
        )
    ratings_jumps[dm] += actions[dm, :, 5] * above_deadzone * (2 + cosine_sim)

    # For boost, just match the value
    boost = boost.values
    boost_diff = np.abs(actions[:, :, 6] - boost.reshape(-1, 1))
    ratings_grounded -= boost_diff
    ratings_aerial -= boost_diff

    # Match throttle, steer and handbrake for grounded actions. Hard to quantify importance of each so just use binary
    mask = ~pressing_jump
    throttle = throttle.values
    steer = steer.values
    handbrake = handbrake.values
    tsh = np.stack([throttle, steer, handbrake], axis=-1)
    tsh[mask & (boost > 0), 0] = np.nan  # Don't count throttle when boosting
    ratings_grounded[mask] -= np.sum(np.abs(actions[mask][:, :, [0, 1, 7]]
                                            - tsh[mask].reshape(-1, 1, 3)) > 0.5)

    # For aerial actions, match pitch, yaw, and roll
    mask = ((~pressing_jump & (dodge_active.values <= 0))  # Don't apply while jumping or dodging
            | (jm & (jumping.diff().values == 0)))  # Except for held down jump button
    ratings_aerial[mask] -= np.linalg.norm(actions[mask, :, 2:5] - pyr[mask].reshape(-1, 1, 3), axis=-1)

    return controls_df, np.stack([ratings_jumps, ratings_grounded, ratings_aerial])


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
