"""Read existing JSONL as named training metrics; --follow displays a live run."""
import argparse
import json
from pathlib import Path
import time

from training_display import format_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    model = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))["recipe"]["model"]
    positions = {"train.jsonl": 0, "valid.jsonl": 0}
    while True:
        rows = []
        for name in positions:
            path = args.run_dir / name
            if not path.exists():
                continue
            if path.stat().st_size < positions[name]:
                positions[name] = 0  # A resumed run may trim uncheckpointed log entries.
            with path.open("rb") as stream:
                stream.seek(positions[name])
                while True:
                    line = stream.readline()
                    if not line.endswith(b"\n"):
                        break
                    positions[name] = stream.tell()
                    rows.append(json.loads(line))
        for row in sorted(rows, key=lambda row: (row["completed_updates"], "native_returns" not in row)):
            print(format_row(model, row), flush=True)
        if not args.follow:
            break
        time.sleep(2)


if __name__ == "__main__":
    main()
