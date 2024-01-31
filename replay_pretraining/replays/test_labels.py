import os
import random

import numpy as np
import pandas as pd
import rlgym
from rlgym.utils import StateSetter
from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import GameState
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.state_setters.wrappers import CarWrapper
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition

BUTTONS = ['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake']


class ReplaySetter(StateSetter):
    def __init__(self, replay_paths: list, random_frames=True):
        super().__init__()
        self.replay_paths = replay_paths
        self.action_sequence = None
        self.random_frames = random_frames

        file_segment_pairs = []
        for replay in replay_paths[:10]:
            for gameplay_segment in os.listdir(replay):
                gamestates = pd.read_parquet(os.path.join(replay, gameplay_segment, "gamestates.parquet"))
                gamestates = [GameState(row.tolist()) for row in gamestates.fillna(0.).replace(np.inf, 300).values]
                idm_actions = pd.read_parquet(os.path.join(replay, gameplay_segment, "idm_actions.parquet"))
                file_segment_pairs.append((os.path.basename(replay), gameplay_segment, gamestates, idm_actions))
        self.file_segment_pairs = file_segment_pairs
        self.pair_idx = None
        self.state_idx = None

    def reset(self, state_wrapper: StateWrapper):
        if self.pair_idx is None:
            self.pair_idx = 0
            self.state_idx = 0
        replay, gameplay_segment, gamestates, actions = self.file_segment_pairs[self.pair_idx]

        if self.state_idx >= len(gamestates):
            self.pair_idx = (self.pair_idx + 1) % len(self.file_segment_pairs)
            replay, gameplay_segment, gamestates, actions = self.file_segment_pairs[self.pair_idx]
            self.state_idx = 0

        # gamestates = [GameState(row.tolist()) for row in gamestates.values]
        idx = self.state_idx
        gamestates = gamestates[idx:]
        actions = actions.iloc[idx:]

        print(f"State is from replay {replay}, segment {gameplay_segment}, at timestep {idx}")

        gamestate = gamestates[0]
        gamestate.players = sorted(gamestate.players, key=lambda p: p.team_num)
        b = o = 0
        self.action_sequence = np.zeros((len(gamestates), len(gamestate.players), 8), dtype=np.float32)
        for i, player in enumerate(gamestate.players):
            car_id = player.car_id
            if player.team_num == BLUE_TEAM:
                player.car_id = StateWrapper.BLUE_ID1 + b
                b += 1
            else:
                player.car_id = StateWrapper.ORANGE_ID1 + o
                o += 1
            self.action_sequence[:, i] = actions[[f"{car_id}/{b}" for b in BUTTONS]].values
        state_wrapper._read_from_gamestate(gamestate)  # noqa
        self.state_idx += 30


def main():
    base_folder = r"D:\rokutleg\behavioral-cloning\labeled\2v2"
    replays = os.listdir(base_folder)
    replays = [os.path.join(base_folder, replay) for replay in replays]
    setter = ReplaySetter(replays)
    env = rlgym.make(game_speed=1, spawn_opponents=True, team_size=3, state_setter=setter, tick_skip=4,
                     terminal_conditions=[TimeoutCondition(30), GoalScoredCondition()])

    while True:
        env.reset()
        done = False
        action_sequence = setter.action_sequence
        i = 0
        while not done:
            obs, reward, done, _ = env.step(action_sequence[i])
            i += 1
            if i >= len(action_sequence):
                break


if __name__ == '__main__':
    main()
