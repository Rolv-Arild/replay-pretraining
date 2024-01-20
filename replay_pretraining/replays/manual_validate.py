import json

import numpy as np

from replay_pretraining.replays.label_replays import ReplayLabeler
from replay_pretraining.replays.replays import load_parsed_replay


def convert_input_map(inputs):
    throttle = inputs["Throttle"]
    steer = inputs["Steer"]
    pitch = inputs["Pitch"]
    yaw = inputs["Yaw"]
    roll = inputs["Roll"]
    jump = inputs["Jump"]
    boosting = inputs["HoldingBoost"]
    handbrake = inputs["Handbrake"]
    return np.array([throttle, steer, pitch, yaw, roll, jump, boosting, handbrake])


def plot_action_grid(replay, actions):
    import matplotlib.pyplot as plt

    throttle_replay = next(iter(replay.players.values()))["throttle"]
    throttle_actions = 255 * (actions[:, 0] + 1) / 2
    grid = np.zeros((256, 256))
    for a, b in zip(throttle_replay, throttle_actions):
        if np.isnan(a) or np.isnan(b):
            continue
        a = round(a)
        b = round(b)
        grid[a, b] += 1
    grid = np.log1p(grid)
    plt.imshow(grid)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # model = torch.jit.load("idm-model.pt").cuda()
    # manual_validate()

    with open("test_replays/2024.1.13-1.17.38.json") as f:
        actions_dict = json.load(f)
    true_actions = np.array([convert_input_map(a["playerInputsMap"][0]["inputs"]) for a in actions_dict])
    replay = load_parsed_replay("./test_replays/2024.1.13-1.17.38")
    labeler = ReplayLabeler("models/idm-model-dainty-durian-356.pt", 2)

    for df, actions in labeler.label_replay(replay):
        debug = True

    # make_bc_dataset(r"D:\rokutleg\parsed\2021-electrum-replays\ranked-doubles",
    #              r"D:\rokutleg\electrum-dataset")
