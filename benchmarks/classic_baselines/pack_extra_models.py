"""Package three isolated CEM-repository baselines for the existing cloud bundle."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
MODELS = ("moel", "mime", "empdg")


def add_json(archive, name, value):
    content = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path,
                        default=HERE / "workspace/preparation_20261026_v2/cem_original")
    parser.add_argument("--output", type=Path, default=HERE.parents[1] /
                        "results/benchmarks/classic_baselines/baseline_extra_models_20261104.tar.gz")
    args = parser.parse_args()
    version = subprocess.check_output(["git", "-c", "safe.directory=" + HERE.parents[1].as_posix(),
                                       "rev-parse", "HEAD"], cwd=str(HERE), text=True).strip()
    source = args.source_workspace
    provenance = json.loads((source / "_baseline_provenance.json").read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        tools = ("train_baseline.py", "training_display.py", "training_state.py", "native_runtime.py",
                 "native_generation.py", "score_records.py", "audit_data.py", "build_data_bundle.py",
                 "audit_sources.py", "show_log.py", "pack_extra_models.py")
        for name in tools:
            path = HERE / name
            archive.add(path, "baseline_training_v1/" + path.name)
        for pattern in ("configs/*.json", "*.sh", "*.md"):
            for path in sorted(HERE.glob(pattern)):
                archive.add(path, "baseline_training_v1/" + path.relative_to(HERE).as_posix())
        add_json(archive, "baseline_training_v1/release.json", {"git_commit": version})
        # Source isolation, not weight/data backups; reuse the same prepared CEM source.
        for model in MODELS:
            for path in sorted((source / "src").rglob("*.py")):
                archive.add(path, "workspace/" + model + "/" + path.relative_to(source).as_posix())
            add_json(archive, "workspace/" + model + "/_baseline_provenance.json",
                     {"source": "cem", "source_commit": provenance["source_commit"],
                      "native_model": model, "implementation": "CEM repository baseline"})
    print(args.output.resolve())


if __name__ == "__main__":
    main()
