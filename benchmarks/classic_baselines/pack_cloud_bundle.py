"""Package prepared sources and shared data, excluding weights and training logs."""
import argparse
import io
import json
from pathlib import Path
import tarfile

from audit_data import encoded
from audit_sources import sha256

HERE = Path(__file__).resolve().parent


def collect_files(workspace, data_dir):
    files = {}
    for path in HERE.iterdir():
        if path.is_file() and (path.suffix in (".py", ".json", ".md") or path.name in (".gitattributes", ".gitignore")):
            files["tools/" + path.name] = path
    for path in (HERE / "tests").glob("*.py"):
        files["tools/tests/" + path.name] = path
    for model in ("case_original", "multi_trs", "cem_original", "kemp"):
        root = workspace / model
        provenance = root / "_baseline_provenance.json"
        meta = json.loads(provenance.read_text(encoding="utf-8"))
        files["workspace/" + model + "/_baseline_provenance.json"] = provenance
        for name, expected in meta["prepared_files_sha256"].items():
            path = root / name
            if sha256(path) != expected:
                raise ValueError("Prepared source changed: " + str(path))
            files["workspace/" + model + "/" + name] = path
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256(data_dir / "shared_data.pkl") != manifest["bundle_sha256"]:
        raise ValueError("Data bundle changed")
    for name in ("manifest.json", "shared_data.pkl", "assets/NRCDict.json", "canonical/train.jsonl", "canonical/valid.jsonl", "canonical/test.jsonl"):
        files["data/" + name] = data_dir / name
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-file", type=Path, help="Fingerprint only; GloVe is not copied into this archive")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".json").exists():
        parser.error("Use a fresh archive path")
    files = collect_files(args.workspace_root.resolve(), args.data_dir.resolve())
    manifest = {"schema_version": 1, "scope": "four_model_preparation_and_inference_smoke_only",
                "formal_training_ready": False, "weights_included": False,
                "files_sha256": {name: sha256(path) for name, path in sorted(files.items())}}
    if args.embedding_file:
        manifest["external_glove"] = {"filename": args.embedding_file.name, "sha256": sha256(args.embedding_file),
                                      "included_in_archive": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(args.output), "w:gz", compresslevel=3) as archive:
        for name, path in sorted(files.items()):
            if path.is_symlink() or not path.is_file():
                raise ValueError("Regular source/data files required: " + str(path))
            archive.add(str(path), arcname="classic_baselines_bundle/" + name, recursive=False)
        payload = encoded(manifest) + b"\n"
        info = tarfile.TarInfo("classic_baselines_bundle/bundle_manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    report = {"archive": args.output.name, "archive_sha256": sha256(args.output),
              "archive_bytes": args.output.stat().st_size, "file_count": len(files), **manifest}
    args.output.with_suffix(args.output.suffix + ".json").write_bytes(encoded(report) + b"\n")
    print(json.dumps({"archive": str(args.output), "archive_sha256": report["archive_sha256"],
                      "archive_bytes": report["archive_bytes"], "file_count": len(files), "weights_included": False}))


if __name__ == "__main__":
    main()
