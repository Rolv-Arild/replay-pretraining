import os
import subprocess
from argparse import ArgumentParser

from tqdm import tqdm

CARBALL_COMMAND = 'carball.exe -i "{}" -o "{}" parquet'

ENV = os.environ.copy()
ENV["NO_COLOR"] = "1"


def process_replay(replay_path, output_folder, skip_existing=True):
    folder, fn = os.path.split(replay_path)
    replay_name = fn.replace(".replay", "")
    processed_folder = os.path.join(output_folder, replay_name)
    if os.path.isdir(processed_folder) and len(os.listdir(processed_folder)) > 0:
        if skip_existing:
            return
        else:
            os.rmdir(processed_folder)
    os.makedirs(processed_folder, exist_ok=True)

    with open(os.path.join(processed_folder, "carball.o.log"), "w", encoding="utf8") as stdout_f:
        with open(os.path.join(processed_folder, "carball.e.log"), "w", encoding="utf8") as stderr_f:
            return subprocess.run(
                CARBALL_COMMAND.format(replay_path, processed_folder),
                stdout=stdout_f,
                stderr=stderr_f,
                env=ENV
            )


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)

    args = parser.parse_args()

    replay_folder = os.path.join(args.input_folder)
    replay_paths = [os.path.join(dp, f)
                    for dp, dn, fn in os.walk(replay_folder)
                    for f in fn
                    if f.endswith(".replay")]
    it = tqdm(sorted(replay_paths), "Parsing replays")
    for replay_path in it:
        parsed_folder = os.path.dirname(replay_path.replace(args.input_folder, args.output_folder))
        process_replay(replay_path, parsed_folder)

        replay_id = os.path.split(replay_path)[-1].replace(".replay", "")
        it.set_postfix_str(replay_id)


if __name__ == '__main__':
    main()
