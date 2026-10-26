"""Full smoke checkpoint state; upstream Noam.state_dict only contains Adam."""
import copy
import hashlib
import random

import numpy as np
import torch

NOAM_FIELDS = ("_step", "_rate", "warmup", "factor", "model_size")


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            digest.update(b"tensor" + str(item.dtype).encode() + str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"numpy" + str(item.dtype).encode() + str(item.shape).encode() + item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for entry in item:
                visit(entry)
            digest.update(b"]")
        else:
            digest.update((type(item).__name__ + ":" + repr(item) + ";").encode())
    visit(value)
    return digest.hexdigest()


def capture(runtime, completed_updates):
    model = runtime.model
    optimizer = model.optimizer
    raw = getattr(optimizer, "optimizer", optimizer)
    auxiliary = {"word_freq": cpu_copy(getattr(model, "word_freq", None)), "modules": {}}
    for name, module in model.named_modules():
        values = {key: cpu_copy(getattr(module, key)) for key in ("true_dist",)
                  if hasattr(module, key)}
        # Dynamic registered buffer: CASE/CEM create criterion.weight during training.
        if "weight" in module._buffers:
            values["weight"] = cpu_copy(module.weight)
        if values:
            auxiliary["modules"][name] = values
    return {"model": cpu_copy(model.state_dict()), "optimizer": cpu_copy(raw.state_dict()),
            "noam": {key: getattr(optimizer, key) for key in NOAM_FIELDS if hasattr(optimizer, key)},
            "auxiliary": auxiliary,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state().clone(),
                    "cuda": cpu_copy(torch.cuda.get_rng_state_all()) if runtime.device.startswith("cuda") else []},
            "completed_updates": completed_updates, "training": model.training,
            "pretrain": bool(getattr(runtime.config, "pretrain", False))}


def protocol(runtime, glove_sha256):
    return {"model_name": runtime.name, "source_commit": runtime.provenance["source_commit"],
            "prepared_sources_sha256": runtime.provenance["prepared_files_sha256"],
            "target_encoding_sha256": runtime.manifest["target_encoding_sha256"],
            "shared_data_sha256": runtime.manifest["bundle_sha256"], "glove_sha256": glove_sha256,
            "trained": True, "scope": "training_smoke_only", "formal_training": False}


def restore(runtime, checkpoint, glove_sha256):
    # Check provenance BEFORE modifying the new model or optimizer.
    if checkpoint.get("baseline_protocol") != protocol(runtime, glove_sha256):
        raise ValueError("Training checkpoint provenance/scope mismatch")
    saved = checkpoint["training_state"]
    model = runtime.model
    modules = dict(model.named_modules())
    for name, values in saved["auxiliary"]["modules"].items():
        for key, value in values.items():
            setattr(modules[name], key, value.to(runtime.device).clone() if torch.is_tensor(value) else copy.deepcopy(value))
    if saved["auxiliary"]["word_freq"] is not None:
        model.word_freq = saved["auxiliary"]["word_freq"].copy()
    model.load_state_dict(saved["model"], strict=True)
    optimizer = model.optimizer
    raw = getattr(optimizer, "optimizer", optimizer)
    raw.load_state_dict(saved["optimizer"])
    for key, value in saved["noam"].items():
        setattr(optimizer, key, value)
    runtime.config.pretrain = saved["pretrain"]
    model.train(saved["training"])
    rng = saved["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    actual = capture(runtime, saved["completed_updates"])
    if fingerprint(actual) != fingerprint(saved):
        raise ValueError("Full training state did not restore bitwise")
    return saved["completed_updates"]


def compare(actual, expected, atol=1e-6, rtol=1e-5):
    """Exact structure/discrete state; explicit tolerance only for floating tensors."""
    result = {"passed": True, "bitwise_equal": fingerprint(actual) == fingerprint(expected),
              "max_abs_error": 0.0, "atol": atol, "rtol": rtol, "mismatches": []}
    def walk(a, b, path):
        ok = True
        if torch.is_tensor(a) and torch.is_tensor(b):
            ok = a.shape == b.shape and a.dtype == b.dtype
            if ok and a.is_floating_point():
                delta = (a.double() - b.double()).abs()
                result["max_abs_error"] = max(result["max_abs_error"], delta.max().item() if delta.numel() else 0)
                ok = torch.allclose(a, b, atol=atol, rtol=rtol)
            elif ok:
                ok = torch.equal(a, b)
        elif isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for key in a:
                walk(a[key], b[key], path + "/" + str(key))
            return
        elif isinstance(a, (list, tuple)) and type(a) is type(b) and len(a) == len(b):
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + "/" + str(index))
            return
        else:
            ok = fingerprint(a) == fingerprint(b)
        if not ok:
            result["passed"] = False
            result["mismatches"].append(path)
    walk(actual, expected, "state")
    return result
