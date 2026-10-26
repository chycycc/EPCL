"""Serial four-model training smoke: two fresh processes per model, hard step cap."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from glove_audit import verify_file

HERE = Path(__file__).resolve().parent
MODELS = ("case_original", "multi_trs", "cem_original", "kemp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace-root", "data-dir", "output-dir", "embedding-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reference-environment", type=Path)
    parser.add_argument("--nltk-data", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.output_dir.exists() or not 1 <= args.batch_size <= 4:
        parser.error("Use a fresh output directory; batch size must be 1..4")
    if args.device == "cuda" and not args.reference_environment:
        parser.error("A measured CASE-EPCL environment reference is required")
    glove_hash = verify_file(args.embedding_file)
    args.output_dir.mkdir(parents=True)
    report = {"scope": "training_smoke_only", "formal_training_started": False, "all_passed": False,
              "glove_sha256": glove_hash, "device": args.device, "batch_size": args.batch_size, "models": {}}
    def save():
        (args.output_dir / "queue.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    for model in MODELS:
        entry = report["models"][model] = {"passed": False, "workers": {}}
        for phase in ("initial", "resume"):
            output = args.output_dir / model / phase
            initial = args.output_dir / model / "initial"
            command = [sys.executable, "-u", str(HERE / "native_train_smoke.py"), "--model", model,
                       "--workspace", str(args.workspace_root / model), "--data-dir", str(args.data_dir),
                       "--output-dir", str(output), "--device", args.device,
                       "--embedding-file", str(args.embedding_file), "--batch-size", str(args.batch_size)]
            for flag, value in (("--reference-environment", args.reference_environment), ("--nltk-data", args.nltk_data)):
                if value:
                    command += [flag, str(value)]
            if phase == "resume":
                command += ["--resume-checkpoint", str(initial / "training_smoke_checkpoint.pt"),
                            "--expected-continuation", str(initial / "expected_continuation.pt")]
            print("Training smoke " + model + "/" + phase, flush=True)
            started = time.monotonic()
            with (args.output_dir / (model + "_" + phase + ".log")).open("wb") as log:
                # Authors' CASE/KEMP readers use open() without encoding. UTF-8 is
                # explicit for Windows parity with the Ubuntu cloud locale.
                environment = dict(os.environ, PYTHONUTF8="1")
                finished = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            worker = {"command": command, "exit_code": finished.returncode, "elapsed_seconds": time.monotonic() - started}
            entry["workers"][phase] = worker
            if finished.returncode == 0:
                meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
                worker.update(passed=meta["passed"], actual_optimizer_updates=meta["actual_optimizer_updates"],
                              glove_matched_rows=meta["glove"]["matched_rows"], pid=meta["pid"],
                              target_encoding_sha256=meta["target_encoding_sha256"])
                if phase == "resume":
                    worker.update(checkpoint_restore_bitwise_equal=meta["checkpoint_restore_bitwise_equal"],
                                  continuation_comparison=meta["continuation_comparison"],
                                  objective_comparison=meta["objective_comparison"],
                                  valid_observation_state_unchanged=meta["valid_observation_state_unchanged"])
            else:
                worker["passed"] = False
            save()
            if not worker["passed"]:
                print("FAIL: " + model + "/" + phase + "; inspect log", flush=True)
                return 1
        initial, resume = entry["workers"]["initial"], entry["workers"]["resume"]
        expected_updates = 5 if model == "case_original" else 4
        entry["actual_optimizer_updates"] = initial["actual_optimizer_updates"] + resume["actual_optimizer_updates"]
        entry["passed"] = (entry["actual_optimizer_updates"] == expected_updates and initial["pid"] != resume["pid"]
                           and resume["checkpoint_restore_bitwise_equal"] and resume["continuation_comparison"]["passed"]
                           and resume["valid_observation_state_unchanged"])
        save()
        if not entry["passed"]:
            return 1
    report["all_passed"] = all(e["passed"] for e in report["models"].values())
    report["actual_optimizer_updates"] = sum(e["actual_optimizer_updates"] for e in report["models"].values())
    save()
    print(json.dumps({"all_passed": report["all_passed"], "actual_optimizer_updates": report["actual_optimizer_updates"]}), flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
