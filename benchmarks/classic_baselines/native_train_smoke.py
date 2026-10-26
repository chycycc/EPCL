"""Bounded native training smoke, never an epoch loop or formal training launcher."""
import argparse
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from audit_sources import sha256
from glove_audit import verify_file, verify_initialized
from native_runtime import NativeRuntime, readonly_observation
from training_state import capture, compare, fingerprint, protocol, restore

HERE = Path(__file__).resolve().parent
MODELS = ("case_original", "multi_trs", "cem_original", "kemp")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite_tree(value):
    import torch
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite parameter/optimizer state")
    elif isinstance(value, dict):
        for entry in value.values():
            finite_tree(entry)
    elif isinstance(value, (tuple, list)):
        for entry in value:
            finite_tree(entry)


def run_update(runtime, start, batch_size, iteration, phase):
    import torch
    model = runtime.model
    model.train()
    runtime.config.pretrain = phase == "bow_pretrain"
    dataset = runtime.datasets["train"]
    collate = dataset.collate_fn if runtime.collate is None else runtime.collate
    indices = list(range(start, start + batch_size))
    batch = collate([dataset[index] for index in indices])
    before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    raw = getattr(model.optimizer, "optimizer", model.optimizer)
    original_step, original_zero, original_backward = raw.step, raw.zero_grad, torch.autograd.backward
    events, backward_losses, zero_events = [], [], []
    def zero(*args, **kwargs):
        result = original_zero(*args, **kwargs)
        zero_events.append({"none": sum(p.grad is None for p in model.parameters()),
                            "tensor": sum(p.grad is not None for p in model.parameters())})
        if any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters()):
            raise ValueError("Native zero_grad left nonzero gradients")
        return result
    def backward(tensors, *args, **kwargs):
        values = (tensors,) if torch.is_tensor(tensors) else tensors
        for value in values:
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite optimized objective")
            backward_losses.append(float(value.detach()))
        return original_backward(tensors, *args, **kwargs)
    def step(*args, **kwargs):
        active = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        finite_tree(active)
        norm2 = sum(g.detach().double().square().sum().item() for g in active.values())
        if not math.isfinite(norm2) or norm2 <= 0:
            raise ValueError("No finite nonzero backward gradients")
        groups = {"embedding": model.embedding, "classifier": runtime.classifier}
        if phase == "main":
            groups["decoder"] = model.decoder
        group_norms = {name: sum(p.grad.detach().double().square().sum().item()
                                for p in module.parameters() if p.grad is not None) ** 0.5
                       for name, module in groups.items()}
        required = ("embedding",) if phase == "bow_pretrain" else tuple(groups)
        if any(group_norms[name] <= 0 for name in required):
            raise ValueError("Missing required training gradients: " + str(group_norms))
        result = original_step(*args, **kwargs)
        events.append({"gradient_norm": norm2 ** 0.5, "gradient_parameter_count": len(active),
                       "required_group_gradient_norms": group_norms,
                       "lr": [group["lr"] for group in raw.param_groups]})
        return result
    def fail_debugger(*args, **kwargs):
        raise ValueError("Native nonfinite-loss debugger branch reached")
    with patch.object(raw, "step", step), patch.object(raw, "zero_grad", zero), \
            patch.object(torch.autograd, "backward", backward), patch("pdb.set_trace", fail_debugger):
        returned = model.train_one_batch(batch, iteration, train=True)
    if len(events) != 1 or len(backward_losses) != 1 or len(zero_events) != 1:
        raise ValueError("Expected exactly one zero_grad, backward and Adam step")
    finite_tree(model.state_dict())
    finite_tree(raw.state_dict())
    changes = {name: (p.detach().cpu() - before[name]).abs().max().item()
               for name, p in model.named_parameters()}
    if max(changes.values()) == 0 or changes["embedding.lut.weight"] == 0:
        raise ValueError("Optimizer step did not change model/embedding weights")
    if phase == "main":
        classifier_ids = {id(p) for p in runtime.classifier.parameters()}
        if not any(changes[name] > 0 for name, p in model.named_parameters() if id(p) in classifier_ids):
            raise ValueError("Classifier weights did not update")
    numeric = [float(v) for v in (returned if isinstance(returned, tuple) else (returned,))
               if isinstance(v, (float, int)) or type(v).__module__ == "numpy"]
    if not all(math.isfinite(v) for v in numeric):
        raise ValueError("Nonfinite native return values")
    row = {"phase": phase, "iteration": iteration, "train_indices": indices, "actual_batch_size": batch_size,
           "objective": backward_losses[0], "native_numeric_returns": numeric,
           "actual_optimizer_updates": len(events), "backward_calls": len(backward_losses),
           "changed_parameter_count": sum(v > 0 for v in changes.values()),
           "parameter_max_abs_change": max(changes.values()), "embedding_max_abs_change": changes["embedding.lut.weight"],
           "zero_grad_signature": str(inspect.signature(original_zero)), "zero_grad_observation": zero_events[0],
           "noam_step": getattr(model.optimizer, "_step", None), **events[0]}
    print("[Training Smoke Update] " + json.dumps(row, allow_nan=False), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reference-environment", type=Path)
    parser.add_argument("--nltk-data", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resume-checkpoint", type=Path, help="Only this smoke format, exactly one resumed update")
    parser.add_argument("--expected-continuation", type=Path)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.output_dir.exists() or not 1 <= args.batch_size <= 4:
        parser.error("Use a fresh output directory and smoke batch size 1..4")
    if bool(args.resume_checkpoint) != bool(args.expected_continuation):
        parser.error("Resume requires both checkpoint and expected continuation")
    if args.nltk_data:
        os.environ["NLTK_DATA"] = str(args.nltk_data)
    import torch
    torch.set_num_threads(1)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "torch_cuda_runtime": torch.version.cuda, "device_used": args.device,
                   "cloud_runtime_validated": False}
    if args.device == "cuda":
        if not args.reference_environment:
            parser.error("Cloud smoke requires the measured CASE-EPCL environment reference")
        from cloud_preflight import collect, validate
        actual = collect()
        reference = json.loads(args.reference_environment.read_text(encoding="utf-8"))["actual"]
        errors = validate(actual, json.loads((HERE / "cloud_target.json").read_text(encoding="utf-8")), reference)
        if errors:
            parser.error("; ".join(errors))
        environment.update(actual=actual, cloud_runtime_validated=True)
        torch.cuda.reset_peak_memory_stats()
    glove_hash = verify_file(args.embedding_file)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "environment.json", environment)
    runtime = NativeRuntime(args.model, args.workspace, args.data_dir, args.output_dir, args.device,
                            seed=13, random_embeddings=False, embedding_file=args.embedding_file,
                            batch_size=args.batch_size)
    glove = verify_initialized(runtime, args.embedding_file)
    print("[GloVe Verified] " + json.dumps(glove), flush=True)
    metadata = {"schema_version": 1, "model": args.model, "scope": "training_smoke_only",
                "formal_training_started": False, "seed": 13, "precision": "fp32", "accum_steps": 1,
                "batch_size": args.batch_size, "glove_sha256": glove_hash, "glove": glove,
                "effective_native_config": runtime.effective_config, "updates": [], "passed": False,
                "pid": os.getpid(), "target_encoding_sha256": runtime.manifest["target_encoding_sha256"],
                "prepared_sources_sha256": runtime.provenance["prepared_files_sha256"],
                "environment_sha256": sha256(args.output_dir / "environment.json"),
                "adapter_sha256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
                "test_samples": 0, "generation_samples": 0}
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        if checkpoint["batch_size"] != args.batch_size or checkpoint["device"] != args.device:
            raise ValueError("Resume batch/device mismatch")
        if checkpoint["environment_sha256"] != metadata["environment_sha256"]:
            raise ValueError("Resume environment differs")
        if checkpoint["adapter_sha256"] != metadata["adapter_sha256"]:
            raise ValueError("Resume adapter source differs")
        completed = restore(runtime, checkpoint, glove_hash)
        metadata["checkpoint_restore_bitwise_equal"] = True
        metadata["restored_state_sha256"] = fingerprint(checkpoint["training_state"])
        expected = torch.load(args.expected_continuation, map_location="cpu")
        if expected["checkpoint_sha256"] != sha256(args.resume_checkpoint):
            raise ValueError("Expected continuation belongs to another checkpoint")
        del checkpoint
        row = run_update(runtime, 2 * args.batch_size, args.batch_size, 2, "main")
        metadata["updates"].append(row)
        actual = capture(runtime, completed + 1)
        metadata["continuation_comparison"] = compare(actual, expected["training_state"])
        metadata["objective_comparison"] = compare(torch.tensor(row["objective"], dtype=torch.float64),
                                                    torch.tensor(expected["objective"], dtype=torch.float64))
        if not metadata["continuation_comparison"]["passed"] or not metadata["objective_comparison"]["passed"]:
            write_json(args.output_dir / "metadata.json", metadata)
            raise ValueError("Resumed next update differs beyond declared atol=1e-6, rtol=1e-5")
        del expected, actual
        # One Valid sample exercises restored weights without Test selection/generation.
        before = fingerprint(capture(runtime, completed + 1))
        runtime.model.eval()
        with readonly_observation(runtime.model):
            metadata["restored_valid_sample"] = runtime.teacher_record(runtime.batch("valid", 0))
        runtime.model.train()
        metadata["valid_observation_state_unchanged"] = fingerprint(capture(runtime, completed + 1)) == before
        if not metadata["valid_observation_state_unchanged"]:
            raise ValueError("Valid observation changed restored training state")
    else:
        if args.model == "case_original":
            metadata["updates"].append(run_update(runtime, 0, args.batch_size, 0, "bow_pretrain"))
        for iteration in range(2):
            metadata["updates"].append(run_update(runtime, iteration * args.batch_size, args.batch_size, iteration, "main"))
        completed = len(metadata["updates"])
        state = capture(runtime, completed)
        checkpoint_path = args.output_dir / "training_smoke_checkpoint.pt"
        torch.save({"baseline_protocol": protocol(runtime, glove_hash), "training_state": state,
                    "batch_size": args.batch_size, "device": args.device,
                    "environment_sha256": metadata["environment_sha256"],
                    "adapter_sha256": metadata["adapter_sha256"]}, checkpoint_path)
        metadata["saved_state_sha256"] = fingerprint(state)
        metadata["checkpoint_sha256"] = sha256(checkpoint_path)
        del state
        row = run_update(runtime, 2 * args.batch_size, args.batch_size, 2, "main")
        metadata["updates"].append(row)
        torch.save({"training_state": capture(runtime, completed + 1), "objective": row["objective"],
                    "checkpoint_sha256": metadata["checkpoint_sha256"]}, args.output_dir / "expected_continuation.pt")
    metadata["actual_optimizer_updates"] = sum(row["actual_optimizer_updates"] for row in metadata["updates"])
    metadata["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
    metadata["passed"] = True
    write_json(args.output_dir / "metadata.json", metadata)
    print("[Training Smoke PASS] " + json.dumps({"model": args.model, "resumed": bool(args.resume_checkpoint),
                                               "actual_updates": metadata["actual_optimizer_updates"]}), flush=True)


if __name__ == "__main__":
    main()
