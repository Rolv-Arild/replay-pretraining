import math
import os.path
import random
from typing import List, Any

import numpy as np
import rlgym
import torch.jit
from rlgym.utils import ObsBuilder, common_values, StateSetter
from rlgym.utils.common_values import BOOST_LOCATIONS
from rlgym.utils.gamestates import GameState, PlayerData, PhysicsObject
from rlgym.utils.state_setters import StateWrapper, DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from torch.distributions import Categorical

from util import lookup_table

boost_locations = np.array(BOOST_LOCATIONS)


class TimerObs(ObsBuilder):
    POS_STD = 2300  # If you read this and wonder why, ping Rangler in the dead of night.
    ANG_STD = math.pi

    def __init__(self, pad_to=231):
        super().__init__()
        self.boost_timers = None
        self.demo_timers = None
        self.pad_to = pad_to
        self.action_history = np.zeros(80)

    def reset(self, initial_state: GameState):
        self.boost_timers = np.zeros(34)
        self.demo_timers = {}  # TODO

    def pre_step(self, state: GameState):
        self.boost_timers = np.clip(self.boost_timers - ts / 120, 1 / 120, None)
        self.boost_timers[state.boost_pads == 1] = 0
        big = boost_locations[:, 2] > 71
        picked_up = (self.boost_timers == 0) & (state.boost_pads == 0)
        self.boost_timers[picked_up & big] = 10
        self.boost_timers[picked_up & ~big] = 4

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = self.boost_timers[::-1]
        else:
            inverted = False
            ball = state.ball
            pads = self.boost_timers

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               previous_action,
               pads / 10]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend([
                (other_car.position - player_car.position) / self.POS_STD,
                (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
            ])

        obs.extend(allies)
        obs.extend(enemies)
        obs = np.concatenate(obs)  # + [self.action_history])
        obs = np.pad(obs, (0, self.pad_to - obs.shape[0]))
        # self.action_history[8:] = self.action_history[:-8]
        # self.action_history[:8] = previous_action
        return obs

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend([
            rel_pos / self.POS_STD,
            rel_vel / self.POS_STD,
            player_car.position / self.POS_STD,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / self.POS_STD,
            player_car.angular_velocity / self.ANG_STD,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car


class GamemodeSetter(StateSetter):
    def __init__(self, setters):
        self.setters = setters

    def build_wrapper(self, max_team_size: int, spawn_opponents: bool) -> StateWrapper:
        gamemode = random.randint(1, 3)
        return StateWrapper(gamemode, gamemode)

    def reset(self, state_wrapper: StateWrapper):
        gamemode = len(state_wrapper.cars) // 2
        self.setters[gamemode - 1].reset(state_wrapper)


if __name__ == '__main__':
    ts = 4
    env = rlgym.make(game_speed=1, spawn_opponents=True, team_size=3,
                     auto_minimize=False,
                     state_setter=GamemodeSetter([
                         DefaultState(), DefaultState(), DefaultState()
                         # ReplaySetter(np.load("plat+dia+champ+gc+ssl_1v1.npy")),
                         # ReplaySetter(np.load("plat+dia+champ+gc+ssl_2v2.npy")),
                         # ReplaySetter(np.load("plat+dia+champ+gc+ssl_3v3.npy"))
                     ]),
                     obs_builder=TimerObs(),
                     terminal_conditions=[TimeoutCondition(120 * 5 * 60 // ts), GoalScoredCondition()],
                     use_injector=True, tick_skip=ts)

    deterministic = False
    m = 1
    model_paths = ["../bc-model-valiant-puddle-180.pt"] * 3 + ["../bc-model-wise-snowflake-178.pt"] * 3
    try:
        with torch.no_grad():
            while True:
                models = [torch.jit.load(os.path.join("../../models", model_path)).cpu().eval()
                          for model_path in model_paths]

                obs, info = env.reset(return_info=True)
                gamemode = len(info["state"].players) // 2
                done = False
                # next_actions = np.zeros((len(models), 8))
                k = 0
                while not done:
                    out = torch.cat([
                        m * models[i + 3 * (i // (2 * gamemode))](
                            torch.from_numpy(np.expand_dims((obs[i][:231]), 0)).float())
                        for i in range(2 * gamemode)
                    ])

                    state = info["state"]
                    for i, player in enumerate(state.players):
                        if (state.ball.linear_velocity < 1).all() and (player.car_data.linear_velocity < 1).all():
                            out[i, lookup_table[:, 0] == 0] -= 1000

                    dist = Categorical(logits=out)
                    if deterministic:
                        action_indices = out.argmax(dim=-1).numpy()
                    else:
                        action_indices = dist.sample().numpy()

                    if (ts * k) % 120 == 0:
                        print(action_indices, dist.entropy().mean().item())
                    actions = lookup_table[action_indices]
                    for _ in range(4 // ts):
                        obs, reward, done, info = env.step(actions)
                        # np.concatenate((next_actions[:2], actions[2:])))
                        if done:
                            break

                    # for o, a in zip(obs[:], actions):
                    #     o[..., -72:] = o[..., -80:-8]
                    #     o[..., -80:-72] = o[9:17]
                    #     o[..., 9:17] = a
                    # next_actions = actions
                    k += 1
    finally:
        env.close()
