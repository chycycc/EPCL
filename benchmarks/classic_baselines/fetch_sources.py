"""Fetch only pinned public source code into this suite's ignored vendor folder."""
import argparse
import json
from pathlib import Path
import subprocess

from audit_sources import HERE, git, sha256


def fetch(lock, vendor):
    vendor.mkdir(parents=True, exist_ok=True)
    for name, spec in lock["sources"].items():
        root = vendor / name
        if not root.exists():
            subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--depth", "1",
                            "--filter=blob:none", "--sparse", spec["url"], str(root)], check=True)
            if git(root, "rev-parse", "HEAD") != spec["commit"]:
                git(root, "fetch", "--depth", "1", "origin", spec["commit"])
                git(root, "checkout", "--detach", spec["commit"])
            git(root, "sparse-checkout", "set", *spec["sparse_paths"])
        # Existing checkouts are verified, never reset, switched or overwritten.
        if git(root, "rev-parse", "HEAD") != spec["commit"]:
            raise RuntimeError("Existing checkout has another revision: " + str(root))
        if git(root, "remote", "get-url", "origin") != spec["url"]:
            raise RuntimeError("Existing checkout has another remote: " + str(root))
        for path, expected in spec["files_sha256"].items():
            if not (root / path).is_file() or sha256(root / path) != expected:
                raise RuntimeError("Missing or edited source: " + str(root / path))
        print(name + ": verified " + spec["commit"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=HERE / "sources.lock.json")
    parser.add_argument("--vendor", type=Path, default=HERE / "vendor")
    args = parser.parse_args()
    fetch(json.loads(args.lock.read_text(encoding="utf-8")), args.vendor)


if __name__ == "__main__":
    main()
