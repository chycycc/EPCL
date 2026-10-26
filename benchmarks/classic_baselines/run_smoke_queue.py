"""Run the four inference-only smoke workers serially, without starting training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MODELS = ("case_original", "multi_trs", "cem_original", "kemp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reference-environment", type=Path)
    parser.add_argument("--nltk-data", type=Path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-dec-step", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=2)
    args = parser.parse_args()
    for name in ("workspace_root", "data_dir", "output_dir", "reference_environment", "nltk_data"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    if not 1 <= args.limit <= 8:
        parser.error("Smoke count must be 1..8")
    args.output_dir.mkdir(parents=True)
    report = {"schema_version": 1, "scope": "untrained_inference_only_smoke", "training_started": False,
              "device": args.device, "models": {}, "all_passed": False}
    report_path = args.output_dir / "queue.json"
    for model in MODELS:
        command = [sys.executable, "-u", str(HERE / "native_export.py"), "--model", model,
                   "--workspace", str(args.workspace_root / model), "--data-dir", str(args.data_dir),
                   "--output-dir", str(args.output_dir / model), "--device", args.device,
                   "--random-embeddings", "--limit", str(args.limit), "--max-dec-step", str(args.max_dec_step),
                   "--beam-size", str(args.beam_size)]
        for name, value in (("--reference-environment", args.reference_environment), ("--nltk-data", args.nltk_data)):
            if value is not None:
                command += [name, str(value)]
        print("Starting inference smoke: " + model, flush=True)
        started = time.monotonic()
        with (args.output_dir / (model + ".log")).open("wb") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        entry = {"command": command, "exit_code": completed.returncode,
                 "elapsed_seconds": time.monotonic() - started}
        if completed.returncode == 0:
            run = args.output_dir / model
            metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines()]
            entry.update(evaluated_n=metadata["evaluated_n"], checkpoint_roundtrip_exact=metadata["checkpoint_roundtrip_exact"],
                         model_state_unchanged=metadata["model_state_unchanged"], actual_optimizer_updates=metadata["actual_optimizer_updates"],
                         target_encoding_sha256=metadata["target_encoding_sha256"],
                         max_native_ce_abs_error=max(row["native_ce_abs_error"] for row in rows),
                         native_greedy_equal=all(row["native_greedy_equal"] for row in rows),
                         poisoned_target_first_distribution_equal=all(row["poisoned_target_first_distribution_equal"] for row in rows))
            entry["passed"] = (entry["evaluated_n"] == args.limit and entry["checkpoint_roundtrip_exact"]
                               and entry["model_state_unchanged"] and entry["actual_optimizer_updates"] == 0
                               and entry["native_greedy_equal"] and entry["poisoned_target_first_distribution_equal"])
        else:
            entry["passed"] = False
        report["models"][model] = entry
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(model + (": PASS" if entry["passed"] else ": FAIL; inspect its log"), flush=True)
        if not entry["passed"]:
            return 1
    report["same_target_encoding"] = len({entry["target_encoding_sha256"] for entry in report["models"].values()}) == 1
    report["all_passed"] = report["same_target_encoding"] and all(entry["passed"] for entry in report["models"].values())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
