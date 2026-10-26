"""Make isolated source copies with explicit, hash-locked infrastructure fixes.

Does not import upstream code, copy data/weights, install packages or launch training.
"""
import argparse
import difflib
import json
from pathlib import Path

from audit_sources import HERE, audit, sha256


def patch_file(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Patch target must occur exactly once")
    return text.replace(old, new, 1)


def changes(source, path, text):
    if source == "case" and path == "src/models/common.py":
        # np.float was a float64 alias, removed in NumPy 1.24; retain its dtype.
        return patch_file(text, "astype(np.float)", "astype(np.float64)")
    if source in ("case", "cem") and path == "main.py":
        old = "    # Intialization\n    for n, p in model.named_parameters():\n        if p.dim() > 1 and (n != \"embedding.lut.weight\" and config.pretrain_emb):\n            xavier_uniform_(p)"
        new = "    # Initialize fresh training only; constructors restore evaluation weights.\n    if not is_eval:\n        for n, p in model.named_parameters():\n            if p.dim() > 1 and (n != \"embedding.lut.weight\" and config.pretrain_emb):\n                xavier_uniform_(p)"
        return patch_file(text, old, new)
    if path == "src/utils/data/loader.py" and source in ("case", "cem"):
        # Limit replacement to the named validation DataLoader, preserving Train shuffle.
        start = text.index("    data_loader_val = torch.utils.data.DataLoader(")
        end = text.index("    data_loader_tst = torch.utils.data.DataLoader(", start)
        region = patch_file(text[start:end], "shuffle=True", "shuffle=False")
        return text[:start] + region + text[end:]
    if source == "kemp" and path == "code/dataloader.py":
        start = text.index("    data_loader_val = torch.utils.data.DataLoader(")
        end = text.index("    data_loader_tst = torch.utils.data.DataLoader(", start)
        region = patch_file(text[start:end], "shuffle=True", "shuffle=False")
        return text[:start] + region + text[end:]
    return text


def prepare(lock, vendor, workspace):
    checked = audit(lock, vendor)
    if not checked["all_passed"]:
        raise ValueError("Source audit failed")
    if workspace.exists():
        raise ValueError("Use a new workspace; never overwrite an existing experiment")
    workspace.mkdir(parents=True)
    report = {"schema_version": 1, "training_started": False, "cloud_ready": False, "models": {}}
    for model, profile in lock["models"].items():
        source = profile["source"]
        spec = lock["sources"][source]
        dest = workspace / model
        dest.mkdir()
        applied = []
        for name in spec["files_sha256"]:
            src = vendor / source / name
            raw = src.read_bytes()
            # Normalize only the copied UTF-8 source, preserving raw vendor bytes.
            before = raw.decode("utf-8-sig").replace("\r\n", "\n")
            after = changes(source, name, before)
            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(after.encode("utf-8"))
            if before != after:
                diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                 fromfile="a/" + name, tofile="b/" + name))
                applied.append({"path": name, "upstream_sha256": sha256(src),
                                "prepared_sha256": sha256(out), "diff": diff})
        entry = {"source": source, "source_commit": spec["commit"], "native_model": profile["native_model"],
                 "applied_fixes": applied,
                 "prepared_files_sha256": {name: sha256(dest / name) for name in spec["files_sha256"]},
                 "data_installed": False, "model_runtime_checked": False}
        (dest / "_baseline_provenance.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["models"][model] = entry
    (workspace / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=HERE / "sources.lock.json")
    parser.add_argument("--vendor", type=Path, default=HERE / "vendor")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(json.loads(args.lock.read_text(encoding="utf-8")), args.vendor, args.workspace)
    print(json.dumps({"prepared_models": list(report["models"]), "cloud_ready": report["cloud_ready"]}))


if __name__ == "__main__":
    main()
