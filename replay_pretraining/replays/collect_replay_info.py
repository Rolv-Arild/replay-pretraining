import json
import os.path
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import ballchasing as bc
import numpy as np
import tqdm

api = bc.Api(sys.argv[1])


def numerical_rank(rank):
    tier = rank["tier"]
    division = rank["division"]
    if tier == 22:
        return 22
    return tier + division / 5


def find_rank_span(player_id, playlist):
    min_rank = None
    max_rank = None

    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)

    for replay in api.get_replays(player_id=player_id, playlist=playlist,
                                  replay_after=start_date, replay_before=end_date,
                                  count=1000):
        players = [p for t in ("blue", "orange") for p in replay[t]["players"]]
        for player in players:
            pid = player["id"]["platform"] + ":" + player["id"]["id"]
            if pid == player_id:
                if "rank" in player:
                    tier = player["rank"]["tier"]
                    div = player["rank"].get("division", 0)
                    rank = (tier, div)
                    if min_rank is None or rank < min_rank:
                        min_rank = rank
                    if max_rank is None or rank > max_rank:
                        max_rank = rank
                break
    return min_rank, max_rank


def get_replay_info(seasons):
    for season in seasons:
        all_ids = set()
        with (open(f"ranked-replays-season_{season}-ssl.ijson", "w") as info_file):
            start_date = next(api.get_replays(season=season, sort_by="replay-date", sort_dir="asc"))["date"]
            end_date = next(api.get_replays(season=season, sort_by="replay-date", sort_dir="desc"))["date"]

            start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00")) \
                .astimezone(timezone.utc).replace(tzinfo=None)
            end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00")) \
                .astimezone(timezone.utc).replace(tzinfo=None)

            # start_date = datetime(2022, 1, 1)
            # end_date = datetime(2022, 12, 31)

            accepted_replays = {}
            for playlist in bc.Playlist.RANKED:
                # for rank_group in [bc.Rank.BRONZE, bc.Rank.SILVER, bc.Rank.GOLD, bc.Rank.PLATINUM, bc.Rank.DIAMOND,
                #                    bc.Rank.CHAMPION, bc.Rank.GRAND_CHAMPION, (bc.Rank.SUPERSONIC_LEGEND,)][::-1]:
                replay_candidates = []
                for rank_group in [bc.Rank.GRAND_CHAMPION + ("grand-champion", bc.Rank.SUPERSONIC_LEGEND,)]:
                    d = start_date
                    while d <= end_date:
                        replays = list(api.get_replays(replay_after=d,
                                                       replay_before=d + timedelta(1),
                                                       season=season,
                                                       min_rank=rank_group[0],
                                                       max_rank=rank_group[-1],
                                                       playlist=playlist,
                                                       count=10_000))

                        n = 0
                        for replay in replays:
                            if replay["playlist_id"] != playlist:
                                continue
                            if replay["season"] != int(season.replace("f", "")):
                                continue
                            if season.startswith("f") != replay["season_type"].startswith("f"):
                                continue
                            if replay["min_rank"]["id"] not in rank_group or \
                                    replay["max_rank"]["id"] not in rank_group:
                                continue
                            bc_id = replay["id"]
                            rl_id = replay["rocket_league_id"]
                            if bc_id in all_ids or rl_id in all_ids:
                                continue
                            all_ids.add(bc_id)
                            all_ids.add(rl_id)

                            replay_candidates.append(replay)
                            n += 1
                        print(f"{playlist}\t{rank_group[0].replace('-1', '')}\t{str(d.date())}\t{n}")
                        d += timedelta(1)

                highest_recorded_rank = {}
                for replay in replay_candidates:
                    for team in ("blue", "orange"):
                        for player in replay[team].get("players", []):
                            pid = (player["id"]["platform"], player["id"]["id"])
                            if "rank" in player:
                                tier = player["rank"]["tier"]
                                div = player["rank"].get("division", 0)
                                rank = max(highest_recorded_rank.get(pid, (-1, -1)), (tier, div))
                                highest_recorded_rank[pid] = rank

                n = 0
                for replay in replay_candidates:
                    ssl = 0
                    total = 0
                    for team in ("blue", "orange"):
                        players = replay[team].get("players")
                        if players is None:
                            break
                        for player in replay[team]["players"]:
                            pid = (player["id"]["platform"], player["id"]["id"])
                            rank = highest_recorded_rank.get(pid)
                            if rank is not None and rank >= (22, 0):
                                ssl += 1
                            total += 1
                    else:
                        if ssl < total:
                            continue

                        info_file.write(json.dumps(replay) + "\n")
                        n += 1
                accepted_replays[playlist] = n
            print(f"Total accepted replays: {accepted_replays}")


def bin_replays(season):
    with open(f"ranked-replays-season_{season}-ssl.ijson") as f:
        replays = [json.loads(line) for line in f]
    bins = {}
    indices = np.random.RandomState(42).permutation(len(replays))
    for i in indices:
        replay = replays[i]
        playlist = replay["playlist_id"]
        rank = replay["max_rank"]["id"]  # .replace("-1", "").replace("-2", "").replace("-3", "")
        l = bins.setdefault(playlist, {}).setdefault(rank, [])
        l.append(replay["id"])

    print(json.dumps({gamemode: {rank: len(l) for rank, l in rd.items()} for gamemode, rd in bins.items()}, indent=4))
    s = json.dumps(bins, indent=4)
    s = re.sub(r",\n\s*", ", ", s)
    s = re.sub(r"\[\n\s*", "[", s)
    s = re.sub(r"\n\s*]", "]", s)
    with open(f"binned_replay_ids-season_{season}-ssl.json", "w") as f:
        f.write(s)


def download_replays():
    with open("binned_replay_ids.json") as f:
        bins = json.load(f)

    progress = tqdm.tqdm(desc="Replay files", total=3 * 8 * 1000)
    folder = r"D:\rokutleg\replays\2022-ranked-replays"
    for playlist in bins:
        for rank in bins[playlist]:
            bin_folder = os.path.join(folder, playlist, rank)
            os.makedirs(bin_folder, exist_ok=True)
            n = 0
            for replay_id in bins[playlist][rank]:
                try:
                    api.download_replay(replay_id, bin_folder)
                    progress.update()
                    n += 1
                    if n >= 1000:
                        break
                except ValueError:
                    pass


def download_ssl_replays(season):
    with open(f"binned_replay_ids-season_{season}-ssl.json") as f:
        bins = json.load(f)

    def download(replay_id, bin_folder):
        try:
            api.download_replay(replay_id, bin_folder)
            return True
        except ValueError:
            return False

    folder = rf"E:\rokutleg\replays\season-{season}-ssl-replays"
    for playlist in bins:
        bin_folder = os.path.join(folder, playlist)
        os.makedirs(bin_folder, exist_ok=True)
        downloaded_ids = set(f.replace(".replay", "") for f in os.listdir(bin_folder) if not f.startswith("__"))
        replay_ids = sum(bins[playlist].values(), start=[])
        replay_ids = list(set(replay_ids))
        random.shuffle(replay_ids)
        # replay_ids = set(replay_ids)
        progress = tqdm.tqdm(desc=f"Downloading replay files for {playlist}",
                             total=len(replay_ids),
                             initial=len(downloaded_ids))
        for replay_id in replay_ids:
            # if len(downloaded_ids) >= 100_000:
            #     break
            if replay_id in downloaded_ids:
                continue
            file = f"{bin_folder}/{replay_id}.replay"
            if os.path.isfile(file):
                success = True
            else:
                success = download(replay_id, bin_folder)

            if success:
                progress.update()
                downloaded_ids.add(replay_id)
        progress.close()


def filter_replays(season):
    # Move replays to a new folder but keep only the best ones,
    # scored by a combination of number of SSL players and number of pro players as recognized by ballchasing
    with open(f"ranked-replays-season_{season}-ssl.ijson") as f:
        replays = {}
        for line in f:
            replay = json.loads(line)
            replays[replay["id"]] = replay
    scores = {"ranked-duels": {}, "ranked-doubles": {}, "ranked-standard": {}}
    for rid, replay in replays.items():
        ssl = 0
        pro = 0
        for team in ("blue", "orange"):
            players = replay[team].get("players")
            if players is None:
                break
            for player in replay[team]["players"]:
                if player.get("pro", False):
                    pro += 1
                if player.get("rank", {}).get("id", None) == "supersonic-legend":
                    ssl += 1
        else:
            playlist = replay["playlist_id"]
            scores[playlist][rid] = (pro, ssl)
    for playlist in scores:
        scores[playlist] = sorted(scores[playlist].items(), key=lambda x: x[1], reverse=True)

    # Limit the number of replays such that one playlist can only represent up to 50% of all replays,
    # meaning the other two playlists will represent the other 50% combined
    limit = float("inf")
    for playlist in "ranked-duels", "ranked-doubles", "ranked-standard":
        total_others = sum(len(scores[p]) for p in scores if p != playlist)
        limit = min(limit, total_others)

    # Move the relevant replays to a new folder
    from_folder = rf"E:\rokutleg\replays\season-{season}-ssl-replays"
    to_folder = rf"E:\rokutleg\replays\season-{season}-ssl-replays-filtered"
    for playlist in scores:
        os.makedirs(f"{to_folder}/{playlist}", exist_ok=True)

        n = 0
        for rid, _ in scores[playlist]:
            src = f"{from_folder}/{playlist}/{rid}.replay"
            if not os.path.isfile(src):
                continue
            dst = f"{to_folder}/{playlist}/{rid}.replay"
            shutil.copy(src, dst)
            n += 1
            if n >= limit:
                break

        # Add __ballchasing.ijson file
        with open(f"{to_folder}/{playlist}/__ballchasing.ijson", "w") as f:
            for fname in os.listdir(f"{to_folder}/{playlist}"):
                if fname.endswith(".replay"):
                    rid = fname.replace(".replay", "")
                    f.write(json.dumps(replays[rid]) + "\n")


def seconds_to_dhms(seconds):
    # Calculate the number of days, hours, minutes, and seconds
    days, seconds = divmod(seconds, 86400)  # 1 day has 86400 seconds
    hours, seconds = divmod(seconds, 3600)  # 1 hour has 3600 seconds
    minutes, seconds = divmod(seconds, 60)  # 1 minute has 60 seconds

    return f"{days:.0f}d {hours:.0f}h {minutes:.0f}m {seconds:.0f}s"


def download_rlcs_replays():
    folder = r"E:\rokutleg\replays\RLCS"
    os.makedirs(folder, exist_ok=True)

    groups = []
    creators = ("76561199022336078", "76561199225615730")
    for creator in creators:
        for group in api.get_groups(creator=creator):
            name = group["name"].lower()
            if name.startswith("rlcs"):
                # RLStats has Season 1 up to and including 21-22, afterwards RLCS Referee is responsible
                if creator == creators[0] and "rlcs 20" in name and "rlcs 2021" not in name:
                    continue
                if creator == creators[1] and "rlcs 21" in name:
                    continue
                groups.append(group)

    groups = sorted(groups, key=lambda g: ("season" not in g["id"], g["id"]))[::-1]

    def foo(group):
        name = group["name"]
        print(f"Started processing '{name}'")
        t0 = time.time()
        folder_path = os.path.join(folder, group["name"])
        if not os.path.exists(folder_path):
            api.download_group(group["id"], folder_path)
            with open(os.path.join(folder_path, "__ballchasing_info.ijson"), "w") as writer:
                for replay in api.get_group_replays(group["id"]):
                    writer.write(json.dumps(replay) + "\n")
        t1 = time.time()
        print(f"'{name}' done! ({seconds_to_dhms(t1 - t0)})")

    with ThreadPoolExecutor(1) as ex:
        for _ in ex.map(foo, groups):
            pass
    # for group in api.get_groups(creator="76561199022336078"):
    #     if "rlcs" in group["name"].lower():
    #         print(group["name"])
    #         api.download_group(group["id"], os.path.join(folder, group["name"]))
    #         print("Done!")


def produce_ijson(folder):
    with open(folder + "-replays.ijson", "w") as f:
        for root, dirs, files in os.walk(folder):
            for filename in files:
                replay_id = filename.replace(".replay", "")
                replay = api.get_replay(replay_id)
                f.write(json.dumps(replay) + "\n")


def main():
    seasons = ["f12"]
    for season in seasons:
        print(f"Processing season {season}")
        get_replay_info([season])
        bin_replays(season)
        # download_replays()
        download_ssl_replays(season)
        filter_replays(season)
    # download_electrum_replays()
    # download_rlcs_replays()
    # base_path = r"D:\rokutleg\replays"
    # for folder in "2022-electrum-replays", "2022-ranked-replays", "2022-ssl-replays", "RLCS":
    #     path = os.path.join(base_path, folder)
    #     produce_ijson(path)
    #     print(path)


if __name__ == '__main__':
    main()
