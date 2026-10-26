"""Inspect pinned baseline sources without importing any model or CASE modules."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=" + root.resolve().as_posix(),
         "-C", str(root), *args], stderr=subprocess.PIPE,
    ).decode("utf-8").strip()


def parser_defaults(path):
    """Read declared CLI defaults via AST, with no import-time side effects."""
    result = {}
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            name = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if not isinstance(name, str) or not name.startswith("--"):
            continue
        options = {}
        for keyword in node.keywords:
            try:
                options[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                pass
        result[name[2:].replace("-", "_")] = options.get(
            "default", False if options.get("action") == "store_true" else None
        )
    return result


def source_files(root):
    # Vendor binary data, checkpoints and tracked __pycache__ are intentionally excluded.
    return [name for name in git(root, "ls-files").splitlines()
            if (root / name).is_file() and "__pycache__" not in name
            and Path(name).suffix.lower() in (".py", ".sh", ".md", ".txt")]


def audit(lock, vendor, local_cem=None):
    result = {"schema_version": 1, "training_started": False,
              "model_modules_imported": False, "sources": {}, "all_passed": True}
    for name, spec in lock["sources"].items():
        root = vendor / name
        entry = {"expected_commit": spec["commit"], "errors": []}
        try:
            entry["actual_commit"] = git(root, "rev-parse", "HEAD")
            entry["actual_remote"] = git(root, "remote", "get-url", "origin")
            if entry["actual_commit"] != spec["commit"]:
                entry["errors"].append("commit mismatch")
            if entry["actual_remote"] != spec["url"]:
                entry["errors"].append("remote mismatch")
            entry["files"] = {}
            for path, expected in spec["files_sha256"].items():
                file_path = root / path
                actual = sha256(file_path) if file_path.is_file() else None
                entry["files"][path] = actual
                if actual != expected:
                    entry["errors"].append("source bytes mismatch: " + path)
            entry["declared_cli_defaults"] = parser_defaults(root / spec["config_path"])
        except (OSError, subprocess.CalledProcessError, SyntaxError) as exc:
            entry["errors"].append(str(exc))
        entry["passed"] = not entry["errors"]
        result["all_passed"] &= entry["passed"]
        result["sources"][name] = entry
    if local_cem:
        reference = vendor / "cem"
        comparison = {}
        for name in source_files(reference):
            candidate = local_cem / name
            if not candidate.is_file():
                comparison[name] = "missing_local"
            else:
                # Git CRLF conversion must not be mistaken for a semantic source edit.
                upstream = (reference / name).read_bytes().replace(b"\r\n", b"\n")
                local = candidate.read_bytes().replace(b"\r\n", b"\n")
                comparison[name] = "equal_after_newline_normalization" if upstream == local else "different"
        result["local_cem_comparison"] = comparison
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=HERE / "sources.lock.json")
    parser.add_argument("--vendor", type=Path, default=HERE / "vendor")
    parser.add_argument("--local-cem", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.lock.read_text(encoding="utf-8")), args.vendor, args.local_cem)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": result["all_passed"], "output": str(args.output)}))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
