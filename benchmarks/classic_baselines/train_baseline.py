"""Independent native baseline training, deterministic Valid selection and N=500 generation."""
import argparse
import json
import math
import numbers
import os
from pathlib import Path
import platform
import subprocess

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch.utils.data import DataLoader

from native_runtime import NativeRuntime, readonly_observation
from native_generation import generate
from score_records import likelihood, diversity, read_jsonl
from training_state import capture, restore_state, restore_weights

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def save_checkpoint(path, value):
    # Atomic replacement of the active checkpoint, not a backup copy.
    pending = path.with_suffix(".writing")
    torch.save(value, pending)
    os.replace(pending, path)


def code_version():
    release = HERE / "release.json"
    if release.exists():
        return json.loads(release.read_text(encoding="utf-8"))["git_commit"]
    return subprocess.check_output(["git", "-c", "safe.directory=" + HERE.parents[1].as_posix(),
                                    "rev-parse", "HEAD"], cwd=str(HERE), text=True).strip()


class TrainStream:
    """Seeded epoch permutations with an explicit checkpointed batch cursor."""
    def __init__(self, runtime, batch_size, seed, state=None):
        self.dataset = runtime.datasets["train"]
        self.collate = self.dataset.collate_fn if runtime.collate is None else runtime.collate
        self.batch_size, self.seed = batch_size, seed
        self.epoch, self.offset = (state["epoch"], state["offset"]) if state else (0, 0)
        assert 0 <= self.offset <= len(self.dataset)
        self.order = self.permutation()

    def permutation(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    @property
    def completed_epochs(self):
        return self.epoch + int(self.offset == len(self.dataset))

    def next(self):
        if self.offset == len(self.dataset):
            self.epoch += 1
            self.offset = 0
            self.order = self.permutation()
        indices = self.order[self.offset:self.offset + self.batch_size]
        batch = self.collate([self.dataset[index] for index in indices])
        self.offset += len(indices)
        return batch

    def state_dict(self):
        return {"epoch": self.epoch, "offset": self.offset}


def ordered_loader(runtime, split, batch_size):
    dataset = runtime.datasets[split]
    collate = dataset.collate_fn if runtime.collate is None else runtime.collate
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=0, collate_fn=collate, generator=torch.Generator().manual_seed(13))


def valid_ppl(runtime, batch_size):
    """Match all four native evaluators: exp(mean(batch token CE)), not mean(batch PPL)."""
    model = runtime.model
    was_training = model.training
    model.eval()
    losses, samples = [], 0
    with readonly_observation(model):
        for batch in ordered_loader(runtime, "valid", batch_size):
            result = model.train_one_batch(batch, 0, train=False)
            ce = float(result[3] if runtime.name == "case_original" else result[0])
            assert math.isfinite(ce), "Nonfinite Valid CE"
            losses.append(ce)
            samples += int(batch["target_batch"].size(0))
    model.train(was_training)
    return {"ppl": math.exp(math.fsum(losses) / len(losses)), "n": samples,
            "batches": len(losses), "batch_size": batch_size, "shuffle": False,
            "aggregation": "exp_mean_batch_token_ce"}


def select_best(progress, step, ppl, recipe):
    assert math.isfinite(ppl), "Nonfinite Valid PPL"
    if step < recipe["selection_start_step"]:
        return False
    # <= preserves the authors' tie policy: equal PPL resets patience.
    improved = ppl <= progress["best_ppl"]
    if improved:
        progress.update(best_ppl=ppl, best_step=step, bad_validations=0)
    else:
        progress["bad_validations"] += 1
    return improved


def truncate_resumed_log(path, progress):
    if path.exists():
        rows = read_jsonl(path)
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                if row["completed_updates"] <= progress["completed_updates"]:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def fit(runtime, recipe, output, version, resume=None):
    progress = {"phase": "pretrain" if recipe["pretrain_epochs"] else "main", "main_step": 0,
                "completed_updates": 0, "best_ppl": float("inf"), "best_step": None,
                "bad_validations": 0, "stop_reason": None}
    stream_state = None
    if resume is not None:
        assert resume["format"] == "baseline_training_v1" and resume["recipe"] == recipe, "Resume recipe mismatch"
        restore_state(runtime, resume["training_state"])
        progress = resume["progress"]
        stream_state = resume["train_stream"]
        for name in ("train.jsonl", "valid.jsonl"):
            truncate_resumed_log(output / name, progress)
    stream = TrainStream(runtime, recipe["batch_size"], recipe["seed"], stream_state)
    model = runtime.model
    model.train()

    def persist(improved=False):
        state = capture(runtime, progress["completed_updates"])
        common = {"format": "baseline_training_v1", "recipe": recipe, "git_commit": version,
                  "native_source_commit": runtime.provenance["source_commit"], "progress": dict(progress)}
        if improved:
            save_checkpoint(output / "ppl_best.pt", {**common,
                            "weights": {"model": state["model"], "auxiliary": state["auxiliary"]}})
        save_checkpoint(output / "last.pt", {**common, "training_state": state,
                                               "train_stream": stream.state_dict()})

    while progress["phase"] != "done":
        if progress["phase"] == "pretrain" and stream.completed_epochs >= recipe["pretrain_epochs"]:
            progress["phase"] = "main"
            runtime.config.pretrain = False
            persist()
            print("[Pretrain Complete] " + str(stream.completed_epochs) + " epochs; Adam/Noam retained", flush=True)
        pretrain = progress["phase"] == "pretrain"
        runtime.config.pretrain = pretrain
        result = model.train_one_batch(stream.next(), progress["main_step"], train=True)
        progress["completed_updates"] += 1
        if not pretrain:
            progress["main_step"] += 1
        values = result if isinstance(result, tuple) else (result,)
        numeric = [float(v) for v in values if isinstance(v, numbers.Real)]
        assert all(math.isfinite(v) for v in numeric), "Nonfinite native training result"
        if progress["completed_updates"] % recipe["log_interval"] == 0:
            row = {"phase": progress["phase"], "main_step": progress["main_step"],
                   "completed_updates": progress["completed_updates"], "native_returns": numeric,
                   "noam_step": model.optimizer._step, "lr": model.optimizer._rate,
                   "epoch": stream.epoch, "samples_in_epoch": stream.offset}
            append_json(output / "train.jsonl", row)
            print("[Train] " + json.dumps(row), flush=True)
        if pretrain:
            if progress["completed_updates"] % recipe["checkpoint_interval"] == 0:
                persist()
            continue
        step = progress["main_step"]
        if step % recipe["valid_interval"] == 0:
            model.epoch = step - 1
            metrics = valid_ppl(runtime, recipe["valid_batch_size"])
            improved = select_best(progress, step, metrics["ppl"], recipe)
            if progress["bad_validations"] >= recipe["patience"]:
                progress.update(phase="done", stop_reason="early_stopping")
            elif step >= recipe["max_main_steps"]:
                progress.update(phase="done", stop_reason="budget_limit")
            row = {"main_step": step, "completed_updates": progress["completed_updates"], **metrics,
                   "eligible": step >= recipe["selection_start_step"], "saved_best": improved,
                   "bad_validations": progress["bad_validations"]}
            append_json(output / "valid.jsonl", row)
            print("[Valid] " + json.dumps(row), flush=True)
            persist(improved)
    write_json(output / "training_result.json", {**progress, "recipe": recipe, "git_commit": version,
               "budget_truncated": progress["stop_reason"] == "budget_limit"})
    return progress


def evaluate_best(runtime, output, recipe, version):
    best = torch.load(output / "ppl_best.pt", map_location="cpu")
    assert best["format"] == "baseline_training_v1" and best["recipe"] == recipe
    restore_weights(runtime, best["weights"])
    del best["weights"]
    runtime.config.pretrain = False
    runtime.model.eval()
    canonical = read_jsonl(runtime.data_dir / "canonical/test.jsonl")
    assert len(canonical) == len(runtime.datasets["test"])
    records, generated = [], []
    with (output / "test_records.jsonl").open("w", encoding="utf-8") as test_stream, \
            (output / "generation_500.jsonl").open("w", encoding="utf-8") as gen_stream:
        for index, batch in enumerate(ordered_loader(runtime, "test", 1)):
            gold = canonical[index]
            record = {"sample_id": gold["sample_id"], "emotion_gold": gold["emotion"], **runtime.teacher_record(batch)}
            records.append(record)
            test_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if index < recipe["generation_n"]:
                response = {"sample_id": gold["sample_id"],
                            **generate(runtime, batch, recipe["max_dec_step"], recipe["beam_size"], verify_native=False)}
                generated.append(response)
                gen_stream.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
            if (index + 1) % 100 == 0:
                print("[Test] " + str(index + 1) + "/" + str(len(canonical)), flush=True)
    metrics = {"model": runtime.name, "checkpoint": "ppl_best.pt", "checkpoint_git_commit": best["git_commit"],
               "evaluation_git_commit": version, "checkpoint_main_step": best["progress"]["best_step"],
               "selection_valid_ppl": best["progress"]["best_ppl"],
               "selection_aggregation": "exp_mean_batch_token_ce", "test_batch_size": 1, "shuffle": False,
               "test": likelihood(records), "generation_n": len(generated),
               "generation": {mode: diversity([row[mode + "_tokens"] for row in generated])
                              for mode in ("greedy", "beam")},
               "recipe": recipe, "nll_definition": "target_nll_exclude_PAD_include_EOS_no_smoothing_no_auxiliary",
               "text_normalization": "unicode_NFKC_strip_TreebankWordTokenizer_preserve_case_v1",
               "search": "cumulative_log_probability_keep_finished_beams_v1_length_penalty_0"}
    write_json(output / "eval_metrics.json", metrics)
    print("[Evaluation Complete] " + str(output / "eval_metrics.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "workspace-root", "data-dir", "output-dir", "embedding-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume this run's last.pt")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate this run's ppl_best.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--nltk-data", type=Path)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    assert not (args.resume and args.evaluate_only)
    if args.nltk_data:
        os.environ["NLTK_DATA"] = str(args.nltk_data)
    recipe = json.loads(args.config.read_text(encoding="utf-8"))
    assert recipe["max_main_steps"] >= recipe["selection_start_step"] > 0
    assert recipe["max_main_steps"] % recipe["valid_interval"] == 0
    assert recipe["selection_start_step"] % recipe["valid_interval"] == 0
    assert recipe["batch_size"] > 0 and recipe["valid_batch_size"] > 0 and recipe["patience"] > 0
    assert recipe["generation_n"] == 500 and recipe["accum_steps"] == 1 and recipe["precision"] == "fp32"
    if args.resume or args.evaluate_only:
        assert args.output_dir.is_dir(), "Run directory does not exist"
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    version = code_version()
    torch.set_num_threads(1)
    runtime = NativeRuntime(recipe["model"], args.workspace_root / recipe["model"], args.data_dir,
                            args.output_dir, args.device, seed=recipe["seed"], embedding_file=args.embedding_file,
                            batch_size=recipe["batch_size"], run_config={**recipe, "git_commit": version,
                            "resume": args.resume, "evaluate_only": args.evaluate_only})
    assert runtime.model.optimizer.warmup == recipe["noam_warmup"]
    if not args.resume and not args.evaluate_only:
        write_json(args.output_dir / "run.json", {"recipe": recipe, "git_commit": version,
                   "native_source_commit": runtime.provenance["source_commit"], "effective_config": runtime.effective_config,
                   "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                   "cuda_runtime": torch.version.cuda, "device": args.device,
                                   "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None}})
    if not args.evaluate_only:
        checkpoint = torch.load(args.output_dir / "last.pt", map_location="cpu") if args.resume else None
        fit(runtime, recipe, args.output_dir, version, resume=checkpoint)
        del checkpoint
    evaluate_best(runtime, args.output_dir, recipe, version)


if __name__ == "__main__":
    main()
