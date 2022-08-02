import math
from typing import List, Any

import numpy as np
import rlgym
import torch.jit
from rlgym.utils import ObsBuilder, common_values
from rlgym.utils.common_values import BOOST_LOCATIONS
from rlgym.utils.gamestates import GameState, PlayerData, PhysicsObject
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import RandomState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition

from util import lookup_table

boost_locations = np.array(BOOST_LOCATIONS)


class TimerObs(ObsBuilder):
    POS_STD = 2300  # If you read this and wonder why, ping Rangler in the dead of night.
    ANG_STD = math.pi

    def __init__(self):
        super().__init__()
        self.boost_timers = None
        self.demo_timers = None

    def reset(self, initial_state: GameState):
        self.boost_timers = np.zeros(34)  # TODO
        self.demo_timers = {}  # TODO

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads
        self.boost_timers = np.clip(self.boost_timers - 1 / 120, 0, None)
        big = boost_locations[:, 2] > 71
        picked_up = (self.boost_timers == 0) & (pads == 0)
        self.boost_timers[picked_up & big] = 10
        self.boost_timers[picked_up & ~big] = 4

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               previous_action,
               self.boost_timers / 10]

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
        return np.concatenate(obs)

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


if __name__ == '__main__':
    ts = 4
    env = rlgym.make(game_speed=1, spawn_opponents=True, team_size=2,
                     # state_setter=RandomState(True, True, False),
                     obs_builder=TimerObs(),
                     terminal_conditions=[TimeoutCondition(120 * 30 // ts), GoalScoredCondition()],
                     use_injector=True, tick_skip=ts)
    model = torch.jit.load("bc-model-jolly-star-48.pt").cpu()
    # model.eval()

    try:
        with torch.no_grad():
            while True:
                obs = env.reset()
                done = False
                while not done:
                    out = model(torch.from_numpy(np.stack(obs)).float())
                    action_indices = [np.random.choice(90, p=o.softmax(axis=-1).numpy()) for o in out]
                    # action_indices = (out - 1000 * torch.eye(90)[8]).argmax(axis=-1).numpy()
                    print(action_indices)
                    # actions = [18] * 4
                    actions = lookup_table[action_indices]
                    for _ in range(4 // ts):
                        obs, reward, done, info = env.step(actions)
    finally:
        env.close()
