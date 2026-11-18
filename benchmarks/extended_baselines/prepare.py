"""Prepare pinned source copies and author-provided data, never train."""
import argparse
import json
import io
from pathlib import Path
import shutil
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "iamm": ("https://github.com/zhouzhouyang520/IAMM.git", "8b23e15c3975b8f65107fee05f9fe2be27691c7d"),
    "escm": ("https://github.com/zhouzhouyang520/EmpatheticDialogueGeneration_ESCM.git", "65e729d958f1dada88fb67dfc914829e511c74bc"),
    "seek": ("https://github.com/wlr737/EMNLP2022-SEEK.git", "cd4e5d205d3abc351898a74d39a49c3097c3eb49"),
}


def extract_ed(archive_path, destination):
    with zipfile.ZipFile(archive_path) as archive:
        nested = next((m for m in archive.namelist() if m.endswith("/data.zip")), None)
        if nested:
            with zipfile.ZipFile(io.BytesIO(archive.read(nested))) as data_archive:
                extract_members(data_archive, destination)
        else:
            extract_members(archive, destination)
    assert (destination / "data/ED/dataset_preproc.p").is_file()


def extract_members(archive, destination):
    for member in archive.infolist():
        parts = Path(member.filename).parts
        if "ED" not in parts or member.is_dir():
            continue
        relative = Path(*parts[parts.index("ED"):])
        assert ".." not in relative.parts
        target = destination / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def prepare(output, assets):
    vendor = HERE.parent / "classic_baselines/vendor"
    for family, (url, commit) in SOURCES.items():
        source = vendor / family
        actual = subprocess.check_output(["git", "-c", "safe.directory=" + source.as_posix(),
                                          "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        assert actual == commit
        destination = output / family
        destination.mkdir(parents=True, exist_ok=False)
        fixes = []
        for tree in ("src", "utils"):
            for path in sorted((source / tree).rglob("*.py")):
                relative = path.relative_to(source)
                text = path.read_text(encoding="utf-8-sig")
                changed = text.replace("astype(np.float)", "astype(np.float64)")
                changed = changed.replace("open(config.emb_file, encoding='utf-8').readlines()", "open(config.emb_file, encoding='utf-8')")
                changed = changed.replace("open(config.emb_file).readlines()", "open(config.emb_file, encoding='utf-8')")
                # CPU local evaluation and a single selected cloud GPU.
                if family == "iamm" and relative.as_posix() == "src/models/IAMM/model.py":
                    changed = changed.replace("bz_index.unsqueeze(-1).cuda()", "bz_index.unsqueeze(-1).to(config.device)")
                if family == "escm" and relative.as_posix() == "src/models/ESCM/model.py":
                    changed = changed.replace("torch.from_numpy(np.array(list_)).cuda()", "torch.from_numpy(np.array(list_)).to(config.device)")
                if family == "seek" and relative.as_posix() == "src/utils/data/loader.py":
                    changed = changed.replace("import os, pandas as pd", "import os")
                    changed = changed.replace("def encode(vocab, file):", "def encode(vocab, file):\n    import pandas as pd")
                if text != changed:
                    fixes.append(str(relative))
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(changed)
        (destination / "data").mkdir()
        shutil.copyfile(vendor / "seek/data/NRCDict.json", destination / "data/NRCDict.json")
        if family == "seek":
            shutil.copytree(source / "data/ED", destination / "data/ED")
        else:
            extract_ed(assets / (family + ".zip"), destination)
        (destination / "source.json").write_text(json.dumps(dict(source=family, source_commit=commit,
             url=url, compatibility_fixes=fixes), indent=2), encoding="utf-8")
        print(family + ": " + str(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "workspace")
    parser.add_argument("--assets", type=Path, default=ROOT / "results/benchmarks/modern_baselines/assets")
    args = parser.parse_args()
    prepare(args.output.resolve(), args.assets.resolve())
