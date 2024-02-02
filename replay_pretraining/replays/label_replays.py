import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from replay_pretraining.idm.idm_model import IDMNet
from replay_pretraining.replays.replays import to_rlgym_dfs, get_data_df, load_parsed_replay
from replay_pretraining.utils.util import make_lookup_table, Replay, rolling_window, normalize_quadrant, BUTTONS

MIRROR_ACTIONS = np.array([1, -1, 1, -1, -1, 1, 1, 1])


def mirror_lookup_table_indices(lookup_table):
    mirrored = lookup_table * MIRROR_ACTIONS
    # Find the indices of the mirrored actions in the unmirrored lookup table
    indices = np.zeros(len(lookup_table), dtype=int)
    for i in range(len(lookup_table)):
        idx = np.where(np.all(mirrored[i] == lookup_table, axis=1))[0]
        assert len(idx) == 1, "Lookup table is not symmetric"
        indices[i] = idx.item()
    return indices


class ReplayLabeler:
    def __init__(self, model_path, lut_granularity=1):
        idm_model = IDMNet(1024, 6, 0).cuda()
        state_dict = torch.jit.load(model_path).state_dict()
        idm_model.load_state_dict(state_dict)
        self.idm_model = idm_model.eval()
        bins = 2 * lut_granularity + 1
        self.lookup_table = make_lookup_table(bins, bins, bins, 8 * lut_granularity)
        self.inverse_table_indices = mirror_lookup_table_indices(self.lookup_table)

    def label_replay(self, parsed_replay: Replay):
        with torch.no_grad():
            action_options = torch.from_numpy(self.lookup_table).float().cuda()
            it = to_rlgym_dfs(parsed_replay, self.lookup_table)
            for gamestates_df, controls in it:
                # states = list(to_rlgym(df))
                uids = sorted([col.split("/")[0]
                               for col in gamestates_df.columns
                               if not (col.startswith("ball") or col.startswith("invert")) and "pos_x" in col])
                n_players = len(uids)
                idm_actions = np.zeros((len(gamestates_df), n_players, 8))
                replay_actions = np.zeros((len(gamestates_df), n_players, 8))
                for i, (x, _) in enumerate(get_data_df(gamestates_df, idm_actions)):
                    uid = uids[i]

                    # if lookup_table is None:
                    #     replay_controls = controls[uid]
                    #     lookup_ranks = None
                    # else:
                    replay_controls, lookup_ranks = controls[uid]

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
                    pred_actions = y_hat[0].cpu().numpy()
                    # For the mirrored samples, swap the indices of the logits
                    pred_actions[mirrored] = pred_actions[mirrored][:, self.inverse_table_indices]
                    pred_on_ground, pred_has_jump, pred_has_flip = torch.stack([t.argmax(axis=-1)
                                                                                for t in
                                                                                y_hat[1:]]).cpu().numpy()
                    conf_on_ground, conf_has_jump, conf_has_flip = torch.stack([t.max(axis=-1)[0]
                                                                                for t in
                                                                                y_hat[1:]]).cpu().numpy()

                    gamestates_df[f"{uid}/on_ground"].fillna(pd.Series(pred_on_ground, index=gamestates_df.index),
                                                             inplace=True)
                    gamestates_df[f"{uid}/has_jump"].fillna(pd.Series(pred_has_jump, index=gamestates_df.index),
                                                            inplace=True)
                    gamestates_df[f"{uid}/has_flip"].fillna(pd.Series(pred_has_flip, index=gamestates_df.index),
                                                            inplace=True)

                    pred_on_ground = pred_on_ground.astype(bool)
                    original_preds = self.lookup_table[pred_actions.argmax(axis=-1)]
                    if lookup_ranks is not None:
                        lur = np.zeros(lookup_ranks.shape[1:])
                        lur += lookup_ranks[0]
                        lur[pred_on_ground] += lookup_ranks[1][pred_on_ground]
                        lur[~pred_on_ground] += lookup_ranks[2][~pred_on_ground]
                        action_valid = lur == lur.max(axis=-1, keepdims=True)
                        pred_actions[~action_valid] = -np.inf
                        pred_actions = pred_actions.argmax(axis=-1)

                    if action_options.ndim == 2:
                        ig_actions = self.lookup_table[pred_actions]
                    else:
                        ig_actions = self.lookup_table[range(len(action_options)), pred_actions]

                    # ig_actions[mirrored] *= MIRROR_ACTIONS

                    idm_actions[:, i] = ig_actions
                    replay_actions[:, i] = replay_controls.values
                idm_actions_df = pd.DataFrame(idm_actions.reshape(-1, len(uids) * 8),
                                              columns=[f"{uid}/{b}" for uid in uids for b in BUTTONS],
                                              index=gamestates_df.index)
                replay_actions_df = pd.DataFrame(replay_actions.reshape(-1, len(uids) * 8),
                                                 columns=[f"{uid}/{b}" for uid in uids for b in BUTTONS],
                                                 index=gamestates_df.index)
                yield gamestates_df, replay_actions_df, idm_actions_df


def main(model_path, parsed_replays_folder, output_folder, lut_granularity, overwrite):
    labeler = ReplayLabeler(model_path, lut_granularity)
    folders = [dp for dp, dn, fn in os.walk(parsed_replays_folder) for f in fn if f == "metadata.json"]
    it = tqdm(folders)
    for folder in it:
        try:
            out_folder = folder.replace(parsed_replays_folder, output_folder)
            if os.path.exists(out_folder):
                if overwrite:
                    shutil.rmtree(out_folder)
                else:
                    it.set_postfix_str("Skipping")
                    continue
            try:
                replay = load_parsed_replay(folder)
            except FileNotFoundError:
                continue
            if not (25 < 1 / replay.game.delta.mean() < 35):
                print(f"Skipping {folder} because of delta ({1 / replay.game.delta.mean()} fps)")
                continue
            gameplay_segment = 0
            for gamestates_df, controls_df, idm_actions_df in labeler.label_replay(replay):
                segment_folder = os.path.join(out_folder, f"{gameplay_segment}")
                os.makedirs(segment_folder)
                gamestates_df.to_parquet(os.path.join(segment_folder, "gamestates.parquet"))
                controls_df.to_parquet(os.path.join(segment_folder, "replay_actions.parquet"))
                idm_actions_df.to_parquet(os.path.join(segment_folder, "idm_actions.parquet"))
                gameplay_segment += 1
        except Exception as e:
            print(f"Error in {folder}: {e}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--parsed_replays_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--lut_granularity", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        parsed_replays_folder=args.parsed_replays_folder,
        output_folder=args.output_folder,
        lut_granularity=args.lut_granularity,
        overwrite=args.overwrite
    )
