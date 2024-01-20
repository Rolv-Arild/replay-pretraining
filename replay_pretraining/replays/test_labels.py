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
    def __init__(self, replay_paths: list, random_frames=False):
        super().__init__()
        self.replay_paths = replay_paths
        self.action_sequence = None
        self.random_frames = random_frames

        file_segment_pairs = []
        for replay in replay_paths:
            for gameplay_segment in os.listdir(replay):
                gamestates = pd.read_parquet(os.path.join(replay, gameplay_segment, "gamestates.parquet"))
                idm_actions = pd.read_parquet(os.path.join(replay, gameplay_segment, "idm_actions.parquet"))
                file_segment_pairs.append((os.path.basename(replay), gameplay_segment, gamestates, idm_actions))
        self.file_segment_pairs = file_segment_pairs

    def reset(self, state_wrapper: StateWrapper):
        replay, gameplay_segment, gamestates, idm_actions = random.choice(self.file_segment_pairs)
        gamestates = [GameState(row.tolist()) for row in gamestates.values]
        idx = 0
        if self.random_frames:
            idx = random.randrange(len(gamestates))
            gamestates = gamestates[idx:]
            idm_actions = idm_actions.iloc[idx:]

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
            self.action_sequence[:, i] = idm_actions[[f"{car_id}/{b}" for b in BUTTONS]].values
        state_wrapper._read_from_gamestate(gamestate)  # noqa


def main():
    base_folder = r"E:\rokutleg\labeled\drawn-sun-355"
    replays = os.listdir(base_folder)
    replays = [os.path.join(base_folder, replay) for replay in replays]
    setter = ReplaySetter(replays)
    env = rlgym.make(game_speed=1, spawn_opponents=True, team_size=3, state_setter=setter, tick_skip=4,
                     terminal_conditions=[TimeoutCondition(30 * 3), GoalScoredCondition()])

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
