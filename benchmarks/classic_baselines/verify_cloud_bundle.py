"""Verify an extracted preparation bundle before any cloud smoke command."""
import argparse
import json
from pathlib import Path

from audit_sources import sha256


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    errors = []
    for name, expected in manifest["files_sha256"].items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file() or path.is_symlink():
            errors.append("Invalid or missing file: " + name)
        elif sha256(path) != expected:
            errors.append("Fingerprint mismatch: " + name)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    errors = verify(args.bundle_root)
    print(json.dumps({"bundle_verified": not errors, "errors": errors, "training_started": False}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
