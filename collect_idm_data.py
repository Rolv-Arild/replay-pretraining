import argparse
import os

import numpy as np
import rlgym
import rlgym_sim
from rlgym.utils.state_setters import RandomState
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym_sim.utils import TerminalCondition
from rlgym_sim.utils.gamestates import GameState
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from tqdm import tqdm

from util import random_action, mutate_action

X_MAX = 7000
Y_MAX = 9000
Z_MAX_BALL = 1850
Z_MAX_CAR = 1900
PITCH_MAX = np.pi / 2
YAW_MAX = np.pi
ROLL_MAX = np.pi

TICKS_PER_SECOND = 120


def random_action_hold_duration(p):
    n = 1
    while True:
        x = np.random.uniform()
        if x > p:
            return n
        n += 1


def encode_gamestate(state: GameState):
    state_vals = [0, state.blue_score, state.orange_score]
    state_vals += state.boost_pads.tolist()

    for bd in (state.ball, state.inverted_ball):
        state_vals += bd.position.tolist()
        state_vals += bd.linear_velocity.tolist()
        state_vals += bd.angular_velocity.tolist()

    for p in state.players:
        state_vals += [p.car_id, p.team_num]
        for cd in (p.car_data, p.inverted_car_data):
            state_vals += cd.position.tolist()
            state_vals += cd.quaternion.tolist()
            state_vals += cd.linear_velocity.tolist()
            state_vals += cd.angular_velocity.tolist()
        state_vals += [
            p.match_goals,
            p.match_saves,
            p.match_shots,
            p.match_demolishes,
            p.boost_pickups,
            p.is_demoed,
            p.on_ground,
            p.ball_touched,
            p.has_jump,
            p.has_flip,
            p.boost_amount
        ]
    return state_vals


class CloseToBallCondition(TerminalCondition):
    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        ball_pos = current_state.ball.position
        return any(
            np.linalg.norm(p.car_data.position - ball_pos) <= 300
            for p in current_state.players
        )


class CloseToPlayerCondition(TerminalCondition):
    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        return any(
            np.linalg.norm(p1.car_data.position - p2.car_data.position) <= 300
            for i, p1 in enumerate(current_state.players)
            for p2 in current_state.players[i + 1:]
        )


class RemoveBallState(StateSetter):
    def __init__(self, setter):
        self.setter = setter

    def reset(self, state_wrapper: StateWrapper):
        self.setter.reset(state_wrapper)
        state_wrapper.ball.position[:] = [-1293, 5560, 100]
        state_wrapper.ball.linear_velocity[:] = [0, 0, 0]
        state_wrapper.ball.angular_velocity[:] = [0, 0, 0]


def main(tick_skip, timeout_seconds, prob_continue_action, n_players, include_ball, include_players, use_sim, output_folder):
    terminals = [TimeoutCondition(timeout_seconds * TICKS_PER_SECOND // tick_skip)]
    if not include_ball:
        terminals.append(CloseToBallCondition())
    if not include_players:
        terminals.append(CloseToPlayerCondition())

    state_files = [f"all_scored_replays/states_scores_{mode}.npz" for mode in ["duels", "doubles", "standard"]]
    start_states = np.load(state_files[n_players // 2 - 1])["states"]
    start_states = start_states[~np.isnan(start_states).any(-1)]
    state_setter = RemoveBallState(WeightedSampleSetter(
        [ReplaySetter(start_states), RandomState(ball_rand_speed=True, cars_rand_speed=True, cars_on_ground=False)],
        [0.5, 0.5]
    ))

    if use_sim:
        env = rlgym_sim.make(
            tick_skip=tick_skip,
            terminal_conditions=terminals,
            team_size=(n_players // 2) if n_players > 1 else 1,
            spawn_opponents=n_players > 1,
            state_setter=state_setter,
        )
    else:
        env = rlgym.make(
            game_speed=100,
            auto_minimize=True,
            tick_skip=tick_skip,
            terminal_conditions=terminals,
            team_size=(n_players // 2) if n_players > 1 else 1,
            spawn_opponents=n_players > 1,
            state_setter=state_setter,
        )

    total_steps = 0
    episode = 0
    states, actions, episodes = [], [], []
    it = tqdm()
    while True:
        done = False
        obs, info = env.reset(return_info=True)

        n = 0
        step_actions = [random_action() for _ in range(n_players)]
        while not done:
            if n == 0:
                n = random_action_hold_duration(prob_continue_action)
                step_actions = [mutate_action(a) for a in step_actions]

            state = info["state"]
            state = np.array(encode_gamestate(state))

            states.append(state)
            actions.append(step_actions)
            episodes.append(episode)

            obs, reward, done, info = env.step(step_actions)
            n -= 1

        if n_players * len(states) > 30 * 60 * 60 * 24:
            file_counter = len(os.listdir(output_folder))
            states = np.vstack(states)
            actions = np.vstack(actions)
            episodes = np.array(episodes)
            total_steps += len(states) * n_players
            np.savez_compressed(os.path.join(output_folder, f"states-actions-episodes-{file_counter}.npz"),
                                states=states, actions=actions, episodes=episodes)
            states, actions, episodes = [], [], []
        episode += 1
        it.update()
        it.set_postfix_str(f"total_states={total_steps + n_players * len(states)}, "
                           f"avg_length={(total_steps + n_players * len(states)) / episode:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tick_skip", type=int, default=4)
    parser.add_argument("--timeout_seconds", type=float, default=5)
    parser.add_argument("--prob_continue_action", type=float, default=0.2)
    parser.add_argument("--n_players", type=int, default=6)
    parser.add_argument("--include_ball", action="store_true")
    parser.add_argument("--include_players", action="store_true")
    parser.add_argument("--use_sim", action="store_true")
    parser.add_argument("--output_folder", default=r"E:\rokutleg\idm-states-actions")

    args = parser.parse_args()

    main(
        tick_skip=args.tick_skip,
        timeout_seconds=args.timeout_seconds,
        prob_continue_action=args.prob_continue_action,
        n_players=args.n_players,
        include_ball=args.include_ball,
        include_players=args.include_players,
        use_sim=args.use_sim,
        output_folder=args.output_folder,
    )
