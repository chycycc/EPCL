"""Isolated inference/export worker. There is deliberately no training command."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from audit_data import encoded
from audit_sources import sha256
from native_generation import generate
from native_runtime import NativeRuntime
from score_records import summarize


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("case_original", "cem_original", "multi_trs", "kemp"), required=True)
    parser.add_argument("--workspace", type=Path, required=True, help="That model's prepared source directory")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--random-embeddings", action="store_true", help="Only untrained CPU smoke, never a formal result")
    parser.add_argument("--embedding-file", type=Path)
    parser.add_argument("--reference-environment", type=Path)
    parser.add_argument("--limit", type=int, default=2, help="0 exports the entire Test split from a trained checkpoint")
    parser.add_argument("--max-dec-step", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--nltk-data", type=Path)
    args = parser.parse_args()
    for name in ("workspace", "data_dir", "output_dir", "checkpoint", "embedding_file", "reference_environment", "nltk_data"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    if args.limit < 0 or args.max_dec_step < 1 or args.beam_size < 1:
        parser.error("Invalid sample/search count")
    if args.checkpoint is None and (not args.random_embeddings or args.limit == 0 or args.limit > 8):
        parser.error("No checkpoint: only explicit random-embedding smoke with 1..8 samples is allowed")
    if args.nltk_data:
        os.environ["NLTK_DATA"] = str(args.nltk_data)
    import torch
    torch.set_num_threads(1)
    environment = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
                   "platform": platform.platform(), "device_used": args.device,
                   "nltk": importlib.metadata.version("nltk"), "cloud_runtime_validated": False}
    if args.device == "cuda":
        if args.reference_environment is None:
            parser.error("Cloud exports require a measured CASE-EPCL environment reference")
        from cloud_preflight import collect, validate
        actual = collect()
        target = json.loads((Path(__file__).parent / "cloud_target.json").read_text(encoding="utf-8"))
        reference = json.loads(args.reference_environment.read_text(encoding="utf-8"))["actual"]
        errors = validate(actual, target, reference)
        if errors:
            parser.error("; ".join(errors))
        environment.update(actual=actual, cloud_runtime_validated=True)
    args.output_dir.mkdir(parents=True)
    runtime = NativeRuntime(args.model, args.workspace, args.data_dir, args.output_dir, args.device, args.seed,
                            random_embeddings=args.random_embeddings or args.checkpoint is not None,
                            embedding_file=args.embedding_file)
    raw_optimizer = getattr(runtime.model.optimizer, "optimizer", runtime.model.optimizer)
    optimizer_calls = []
    original_step = raw_optimizer.step
    def observed_step(*step_args, **step_kwargs):
        optimizer_calls.append(True)
        return original_step(*step_args, **step_kwargs)
    raw_optimizer.step = observed_step
    environment_path = args.output_dir / "environment.json"
    environment_path.write_bytes(encoded(environment) + b"\n")
    if args.checkpoint:
        runtime.load_checkpoint(args.checkpoint)
        checkpoint = args.checkpoint
        if not runtime.checkpoint_protocol["trained"] and (args.limit == 0 or args.limit > 8):
            parser.error("Untrained checkpoint: only a 1..8-sample smoke is allowed")
    else:
        checkpoint = args.output_dir / "untrained_smoke_checkpoint.pt"
        before = state_hash(runtime.model)
        torch.save({"model": runtime.model.state_dict(),
                    "baseline_protocol": {"target_encoding_sha256": runtime.manifest["target_encoding_sha256"],
                                          "source_commit": runtime.provenance["source_commit"],
                                          "trained": False}}, checkpoint)
        with torch.no_grad():
            next(runtime.model.parameters()).add_(1)
        runtime.load_checkpoint(checkpoint)
        if state_hash(runtime.model) != before:
            raise ValueError("Checkpoint roundtrip changed native state")
    before = state_hash(runtime.model)
    source_canonical = args.data_dir / "canonical/test.jsonl"
    canonical = [json.loads(line) for line in source_canonical.read_text(encoding="utf-8").splitlines()]
    if args.limit > len(canonical):
        parser.error("Requested count exceeds Test split")
    canonical = canonical[:args.limit] if args.limit else canonical
    canonical_path = args.output_dir / "canonical.jsonl"
    canonical_path.write_bytes(b"".join(encoded(row) + b"\n" for row in canonical))
    records = []
    with (args.output_dir / "records.jsonl").open("wb") as stream:
        for index, gold in enumerate(canonical):
            batch = runtime.batch("test", index)
            expected_target = [runtime.vocabulary["word2index"][word] for word in gold["target"]] + [2]
            if batch["target_batch"][0].tolist() != expected_target:
                raise ValueError("Native target IDs differ from canonical target + EOS")
            row = {"sample_id": gold["sample_id"], "emotion_gold": gold["emotion"],
                   **runtime.teacher_record(batch), **generate(runtime, batch, args.max_dec_step, args.beam_size)}
            stream.write(encoded(row) + b"\n")
            stream.flush()
            records.append(row)
            print(json.dumps({"model": args.model, "completed": index + 1, "total": len(canonical)}), flush=True)
    after = state_hash(runtime.model)
    if before != after:
        raise ValueError("Read-only inference changed model state")
    if optimizer_calls:
        raise ValueError("Inference unexpectedly performed an optimizer update")
    metadata = {"model": args.model, "checkpoint_sha256": sha256(checkpoint),
                "source_commit": runtime.provenance["source_commit"],
                "prepared_sources_sha256": runtime.provenance["prepared_files_sha256"],
                "environment_sha256": sha256(environment_path),
                "target_encoding_sha256": runtime.manifest["target_encoding_sha256"],
                "canonical_sha256": sha256(canonical_path), "shared_data_sha256": runtime.manifest["bundle_sha256"],
                "nll_definition": "target_nll_sum_exclude_pad_include_eos_no_auxiliary_no_smoothing",
                "text_normalization": "unicode_NFKC_strip_TreebankWordTokenizer_preserve_case_v1",
                "generation_protocol": {"max_generated_tokens_including_eos": args.max_dec_step,
                                        "beam_size": args.beam_size, "length_penalty": 0.0,
                                        "first_context_source": "native_reference_free_decoder_greedy",
                                        "search": "cumulative_log_probability_keep_finished_beams_v1"},
                "scope": ("untrained_" + args.device + "_smoke") if not runtime.checkpoint_protocol["trained"] else "checkpoint_inference",
                "trained_checkpoint": runtime.checkpoint_protocol["trained"],
                "effective_native_config": runtime.effective_config,
                "evaluated_n": len(records), "full_test_n": len(runtime.datasets["test"]),
                "seed": args.seed, "precision": "fp32", "training_started": False,
                "model_state_unchanged": True, "checkpoint_roundtrip_exact": args.checkpoint is None,
                "actual_optimizer_updates": len(optimizer_calls),
                "adapter_sha256": {name: sha256(Path(__file__).parent / name) for name in
                                   ("native_export.py", "native_runtime.py", "native_generation.py", "score_records.py")},
                "native_modules": runtime.imported_native_paths}
    (args.output_dir / "metadata.json").write_bytes(encoded(metadata) + b"\n")
    metrics = summarize(records, canonical)
    metrics.update(metadata=metadata)
    (args.output_dir / "metrics.json").write_bytes(encoded(metrics) + b"\n")
    print(json.dumps({"model": args.model, "scope": metadata["scope"], "evaluated_n": len(records),
                      "state_unchanged": True, "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
