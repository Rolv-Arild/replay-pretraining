from typing import Optional, List

import numpy as np


# Methods and classes from RLGym

def quat_to_euler(quat):
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    sinp = 2 * (w * y - z * x)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)

    roll = np.arctan2(sinr_cosp, cosr_cosp)
    if abs(sinp) > 1:
        pitch = np.pi / 2
    else:
        pitch = np.arcsin(sinp)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([-pitch, yaw, -roll])


# From RLUtilities
def quat_to_rot_mtx(quat: np.ndarray) -> np.ndarray:
    w = -quat[0]
    x = -quat[1]
    y = -quat[2]
    z = -quat[3]

    theta = np.zeros((3, 3))

    norm = np.dot(quat, quat)
    if norm != 0:
        s = 1.0 / norm

        # front direction
        theta[0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
        theta[1, 0] = 2.0 * s * (x * y + z * w)
        theta[2, 0] = 2.0 * s * (x * z - y * w)

        # left direction
        theta[0, 1] = 2.0 * s * (x * y - z * w)
        theta[1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
        theta[2, 1] = 2.0 * s * (y * z + x * w)

        # up direction
        theta[0, 2] = 2.0 * s * (x * z + y * w)
        theta[1, 2] = 2.0 * s * (y * z - x * w)
        theta[2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

    return theta


class PhysicsObject(object):
    def __init__(self, position=None, quaternion=None, linear_velocity=None, angular_velocity=None):
        self.position: np.ndarray = position if position is not None else np.zeros(3)

        # ones by default to prevent mathematical errors when converting quat to rot matrix on empty physics state
        self.quaternion: np.ndarray = quaternion if quaternion is not None else np.ones(4)

        self.linear_velocity: np.ndarray = linear_velocity if linear_velocity is not None else np.zeros(3)
        self.angular_velocity: np.ndarray = angular_velocity if angular_velocity is not None else np.zeros(3)
        self._euler_angles: Optional[np.ndarray] = np.zeros(3)
        self._rotation_mtx: Optional[np.ndarray] = np.zeros((3, 3))
        self._has_computed_rot_mtx = False
        self._has_computed_euler_angles = False

    def decode_car_data(self, car_data: np.ndarray):
        """
        Function to decode the physics state of a car from the game state array.
        :param car_data: Slice of game state array containing the car data to decode.
        """
        self.position = car_data[:3]
        self.quaternion = car_data[3:7]
        self.linear_velocity = car_data[7:10]
        self.angular_velocity = car_data[10:]

    def decode_ball_data(self, ball_data: np.ndarray):
        """
        Function to decode the physics state of the ball from the game state array.
        :param ball_data: Slice of game state array containing the ball data to decode.
        """
        self.position = ball_data[:3]
        self.linear_velocity = ball_data[3:6]
        self.angular_velocity = ball_data[6:9]

    def forward(self) -> np.ndarray:
        return self.rotation_mtx()[:, 0]

    def right(self) -> np.ndarray:
        return self.rotation_mtx()[:, 1]

    def left(self) -> np.ndarray:
        return self.rotation_mtx()[:, 1] * -1

    def up(self) -> np.ndarray:
        return self.rotation_mtx()[:, 2]

    def pitch(self) -> float:
        return self.euler_angles()[0]

    def yaw(self) -> float:
        return self.euler_angles()[1]

    def roll(self) -> float:
        return self.euler_angles()[2]

    # pitch, yaw, roll
    def euler_angles(self) -> np.ndarray:
        if not self._has_computed_euler_angles:
            self._euler_angles = quat_to_euler(self.quaternion)
            self._has_computed_euler_angles = True

        return self._euler_angles

    def rotation_mtx(self) -> np.ndarray:
        if not self._has_computed_rot_mtx:
            self._rotation_mtx = quat_to_rot_mtx(self.quaternion)
            self._has_computed_rot_mtx = True

        return self._rotation_mtx

    def serialize(self):
        """
        Function to serialize all the values contained by this physics object into a single 1D list. This can be useful
        when constructing observations for a policy.
        :return: List containing the serialized data.
        """
        repr = []

        if self.position is not None:
            for arg in self.position:
                repr.append(arg)

        if self.quaternion is not None:
            for arg in self.quaternion:
                repr.append(arg)

        if self.linear_velocity is not None:
            for arg in self.linear_velocity:
                repr.append(arg)

        if self.angular_velocity is not None:
            for arg in self.angular_velocity:
                repr.append(arg)

        if self._euler_angles is not None:
            for arg in self._euler_angles:
                repr.append(arg)

        if self._rotation_mtx is not None:
            for arg in self._rotation_mtx.ravel():
                repr.append(arg)

        return repr


class PlayerData(object):
    def __init__(self):
        self.car_id: int = -1
        self.team_num: int = -1
        self.match_goals: int = -1
        self.match_saves: int = -1
        self.match_shots: int = -1
        self.match_demolishes: int = -1
        self.boost_pickups: int = -1
        self.is_demoed: bool = False
        self.on_ground: bool = False
        self.ball_touched: bool = False
        self.has_jump: bool = False
        self.has_flip: bool = False
        self.boost_amount: float = -1
        self.car_data: PhysicsObject = PhysicsObject()
        self.inverted_car_data: PhysicsObject = PhysicsObject()

    def __str__(self):
        output = "****PLAYER DATA OBJECT****\n" \
                 "Match Goals: {}\n" \
                 "Match Saves: {}\n" \
                 "Match Shots: {}\n" \
                 "Match Demolishes: {}\n" \
                 "Boost Pickups: {}\n" \
                 "Is Alive: {}\n" \
                 "On Ground: {}\n" \
                 "Ball Touched: {}\n" \
                 "Has Jump: {}\n" \
                 "Has Flip: {}\n" \
                 "Boost Amount: {}\n" \
                 "Car Data: {}\n" \
                 "Inverted Car Data: {}" \
            .format(self.match_goals,
                    self.match_saves,
                    self.match_shots,
                    self.match_demolishes,
                    self.boost_pickups,
                    not self.is_demoed,
                    self.on_ground,
                    self.ball_touched,
                    self.has_jump,
                    self.has_flip,
                    self.boost_amount,
                    self.car_data,
                    self.inverted_car_data)
        return output


class GameState(object):
    BOOST_PADS_LENGTH = 34
    BALL_STATE_LENGTH = 18
    PLAYER_CAR_STATE_LENGTH = 13
    PLAYER_TERTIARY_INFO_LENGTH = 11
    PLAYER_INFO_LENGTH = 2 + 2 * PLAYER_CAR_STATE_LENGTH + PLAYER_TERTIARY_INFO_LENGTH

    def __init__(self, state_floats: List[float] = None):
        self.game_type: int = 0
        self.blue_score: int = -1
        self.orange_score: int = -1
        self.last_touch: Optional[int] = -1

        self.players: List[PlayerData] = []

        self.ball: PhysicsObject = PhysicsObject()
        self.inverted_ball: PhysicsObject = PhysicsObject()

        # List of "booleans" (1 or 0)
        self.boost_pads: np.ndarray = np.zeros(GameState.BOOST_PADS_LENGTH, dtype=np.float32)
        self.inverted_boost_pads: np.ndarray = np.zeros_like(self.boost_pads, dtype=np.float32)

        if state_floats is not None:
            self.decode(state_floats)

    def decode(self, state_floats: List[float]):
        """
        Decode a string containing the current game state from the Bakkesmod plugin.
        :param state_floats: String containing the game state.
        """
        assert type(state_floats) == list, "UNABLE TO DECODE STATE OF TYPE {}".format(type(state_floats))
        self._decode(state_floats)

    def _decode(self, state_vals: List[float]):
        pads_len = GameState.BOOST_PADS_LENGTH
        p_len = GameState.PLAYER_INFO_LENGTH
        b_len = GameState.BALL_STATE_LENGTH
        start = 3

        num_ball_packets = 1
        # The state will contain the ball, the mirrored ball, every player, every player mirrored, the score for both teams, and the number of ticks since the last packet was sent.
        num_player_packets = int((len(state_vals) - num_ball_packets * b_len - start - pads_len) / p_len)

        ticks = int(state_vals[0])

        self.blue_score = int(state_vals[1])
        self.orange_score = int(state_vals[2])

        self.boost_pads[:] = state_vals[start:start + pads_len]
        self.inverted_boost_pads[:] = self.boost_pads[::-1]
        start += pads_len

        ball_data = state_vals[start:start + b_len]
        self.ball.decode_ball_data(np.asarray(ball_data))
        start += b_len // 2

        inv_ball_data = state_vals[start:start + b_len]
        self.inverted_ball.decode_ball_data(np.asarray(inv_ball_data))
        start += b_len // 2

        for i in range(num_player_packets):
            player = self._decode_player(state_vals[start:start + p_len])
            self.players.append(player)
            start += p_len

            if player.ball_touched:
                self.last_touch = player.car_id

        self.players = sorted(self.players, key=lambda p: p.car_id)  # YOU'RE WELCOME RANGLER, THIS WAS MY INNOVATION.

    def _decode_player(self, full_player_data):
        player_data = PlayerData()
        c_len = GameState.PLAYER_CAR_STATE_LENGTH
        t_len = GameState.PLAYER_TERTIARY_INFO_LENGTH

        start = 2

        car_data = full_player_data[start:start + c_len]
        player_data.car_data.decode_car_data(np.asarray(car_data))
        start += c_len

        inv_state_data = full_player_data[start:start + c_len]
        player_data.inverted_car_data.decode_car_data(np.asarray(inv_state_data))
        start += c_len

        tertiary_data = full_player_data[start:start + t_len]

        player_data.match_goals = int(tertiary_data[0])
        player_data.match_saves = int(tertiary_data[1])
        player_data.match_shots = int(tertiary_data[2])
        player_data.match_demolishes = int(tertiary_data[3])
        player_data.boost_pickups = int(tertiary_data[4])
        player_data.is_demoed = True if tertiary_data[5] > 0 else False
        player_data.on_ground = True if tertiary_data[6] > 0 else False
        player_data.ball_touched = True if tertiary_data[7] > 0 else False
        player_data.has_jump = True if tertiary_data[8] > 0 else False
        player_data.has_flip = True if tertiary_data[9] > 0 else False
        player_data.boost_amount = float(tertiary_data[10])
        player_data.car_id = int(full_player_data[0])
        player_data.team_num = int(full_player_data[1])

        return player_data

    def __str__(self):
        output = "{}GAME STATE OBJECT{}\n" \
                 "Game Type: {}\n" \
                 "Orange Score: {}\n" \
                 "Blue Score: {}\n" \
                 "PLAYERS: {}\n" \
                 "BALL: {}\n" \
                 "INV_BALL: {}\n" \
                 "".format("*" * 8, "*" * 8,
                           self.game_type,
                           self.orange_score,
                           self.blue_score,
                           self.players,
                           self.ball,
                           self.inverted_ball)

        return output
