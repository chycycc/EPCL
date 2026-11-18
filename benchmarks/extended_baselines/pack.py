"""Create the incremental cloud package: tools, pinned sources and author ED caches."""
import io
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    output = ROOT / "results/benchmarks/extended_baselines/extended_baselines_20261118.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(), "rev-parse", "HEAD"], text=True).strip()
    with tarfile.open(output, "w:gz") as archive:
        for pattern in ("*.py", "*.sh", "*.md", "configs/*.json"):
            for path in sorted(HERE.glob(pattern)):
                archive.add(path, "extended_training_v1/" + path.relative_to(HERE).as_posix())
        helpers = ("train_baseline.py", "training_display.py", "training_state.py", "native_runtime.py",
                   "native_generation.py", "score_records.py", "audit_data.py", "build_data_bundle.py", "audit_sources.py")
        for name in helpers:
            archive.add(HERE.parent / "classic_baselines" / name, "extended_training_v1/lib/" + name)
        data_files = dict(iamm=("dataset_preproc.p",), escm=("dataset_preproc.p", "dep_tree.p", "vocab.p", "ConceptNet_VAD_dict.json"),
                          seek=("dataset_preproc.p",))
        for family, names in data_files.items():
            source = HERE / "workspace" / family
            for tree in ("src", "utils"):
                for path in sorted((source / tree).rglob("*.py")):
                    archive.add(path, "workspace_extended/" + family + "/" + path.relative_to(source).as_posix())
            for name in ("source.json", "data/NRCDict.json") + tuple("data/ED/" + n for n in names):
                archive.add(source / name, "workspace_extended/" + family + "/" + name)
        content = json.dumps(dict(git_commit=commit)).encode()
        item = tarfile.TarInfo("extended_training_v1/release.json")
        item.size = len(content)
        archive.addfile(item, io.BytesIO(content))
    print(output)


if __name__ == "__main__":
    main()
