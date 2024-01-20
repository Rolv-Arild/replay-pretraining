import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from replay_pretraining.idm.idm_model import IDMNet
from replay_pretraining.replays.replays import to_rlgym_dfs, get_data_df, load_parsed_replay
from replay_pretraining.utils.util import make_lookup_table, Replay, rolling_window, normalize_quadrant, BUTTONS


class ReplayLabeler:
    def __init__(self, model_path, lut_granularity=1):
        idm_model = IDMNet(1024, 6, 0).cuda()
        state_dict = torch.jit.load(model_path).state_dict()
        idm_model.load_state_dict(state_dict)
        self.idm_model = idm_model.eval()
        bins = 2 * lut_granularity + 1
        self.lookup_table = make_lookup_table(bins, bins, bins, 8 * lut_granularity)

    def label_replay(self, parsed_replay: Replay):
        with torch.no_grad():
            action_options = torch.from_numpy(self.lookup_table).float().cuda()
            it = to_rlgym_dfs(parsed_replay)
            for gamestates_df, controls_df in it:
                # states = list(to_rlgym(df))
                uids = sorted([col.split("/")[0]
                               for col in gamestates_df.columns
                               if not (col.startswith("ball") or col.startswith("invert")) and "pos_x" in col])
                n_players = len(uids)
                actions = np.zeros((len(gamestates_df), n_players, 8))
                for i, (x, _) in enumerate(get_data_df(gamestates_df, actions)):
                    x_rolled = x[rolling_window(np.arange(len(x)), 77, True, True)]
                    mirrored = normalize_quadrant(x_rolled, [np.zeros((len(x), 8))])
                    x_rolled = x_rolled[..., :16]
                    mid = x_rolled.shape[1] // 2
                    x_rolled[:, np.r_[0:mid, mid + 1:x_rolled.shape[1]]] -= x_rolled[:, mid:mid + 1]
                    inp = torch.from_numpy(x_rolled).float().cuda()
                    y_hat = [0.] * 4

                    for _ in range(1):
                        t = self.idm_model(inp, action_options)
                        for j in range(4):
                            y_hat[j] += t[j]

                    # Stack to make gpu->cpu transfer faster
                    pred_actions, pred_on_ground, pred_has_jump, pred_has_flip = torch.stack([t.argmax(axis=-1)
                                                                                              for t in
                                                                                              y_hat]).cpu().numpy()
                    conf_actions, conf_on_ground, conf_has_jump, conf_has_flip = torch.stack([t.max(axis=-1)[0]
                                                                                              for t in
                                                                                              y_hat]).cpu().numpy()

                    if action_options.ndim == 2:
                        pred_actions = action_options[pred_actions].detach().cpu().numpy()
                    else:
                        pred_actions = action_options[range(len(action_options)), pred_actions].detach().cpu().numpy()
                    pred_actions[mirrored] *= np.array([1, -1, 1, -1, -1, 1, 1, 1])

                    uid = uids[i]
                    gamestates_df[f"{uid}/on_ground"] = pred_on_ground == 1
                    gamestates_df[f"{uid}/has_jump"] = pred_has_jump == 1
                    gamestates_df[f"{uid}/has_flip"] = pred_has_flip == 1

                    # state.players[i].on_ground = pred_on_ground[j] == 1
                    # state.players[i].has_jump = pred_has_jump[j] == 1
                    # state.players[i].has_flip = pred_has_flip[j] == 1
                    actions[:, i] = pred_actions
                idm_actions_df = pd.DataFrame(actions.reshape(-1, len(uids) * 8),
                                              columns=[f"{uid}/{b}" for uid in uids for b in BUTTONS])
                yield gamestates_df, controls_df, idm_actions_df


def main(model_path, parsed_replays_folder, output_folder, lut_granularity):
    labeler = ReplayLabeler(model_path, lut_granularity)
    folders = [dp for dp, dn, fn in os.walk(parsed_replays_folder) for f in fn if f == "metadata.json"]
    it = tqdm(folders)
    for folder in it:
        try:
            replay = load_parsed_replay(folder)
        except FileNotFoundError:
            continue
        if not (25 < 1 / replay.game.delta.mean() < 35):
            print(f"Skipping {folder} because of delta ({1 / replay.game.delta.mean()} fps)")
            continue
        gameplay_segment = 0
        for gamestates_df, controls_df, idm_actions_df in labeler.label_replay(replay):
            out_folder = folder.replace(parsed_replays_folder, output_folder)
            out_folder = os.path.join(out_folder, f"{gameplay_segment}")
            os.makedirs(out_folder)
            gamestates_df.to_parquet(os.path.join(out_folder, "gamestates.parquet"))
            controls_df.to_parquet(os.path.join(out_folder, "controls.parquet"))
            idm_actions_df.to_parquet(os.path.join(out_folder, "idm_actions.parquet"))
            gameplay_segment += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--parsed_replays_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--lut_granularity", type=int, default=1)
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        parsed_replays_folder=args.parsed_replays_folder,
        output_folder=args.output_folder,
        lut_granularity=args.lut_granularity
    )

