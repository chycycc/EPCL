"""Record and check the user-confirmed cloud runtime; never start model training."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
PACKAGES = ("torch", "numpy", "scipy", "scikit-learn", "transformers", "tokenizers",
            "nltk", "spacy", "tensorboard", "tensorboardX")


def validate(actual, target, reference=None):
    errors = []
    expected_os = actual.get("os_release", {})
    if expected_os.get("ID") != "ubuntu" or expected_os.get("VERSION_ID") != "22.04":
        errors.append("Expected Ubuntu 22.04")
    if ".".join(actual.get("python", "").split(".")[:2]) != target["python_major_minor"]:
        errors.append("Expected Python " + target["python_major_minor"])
    if actual.get("torch", "").split("+")[0] != target["torch_base_version"]:
        errors.append("Expected PyTorch " + target["torch_base_version"])
    if actual.get("torch_cuda_runtime") != target["torch_cuda_runtime"]:
        errors.append("Expected torch CUDA runtime " + target["torch_cuda_runtime"])
    gpus = actual.get("gpus", [])
    if len(gpus) != target["gpu_count"]:
        errors.append("Expected one visible GPU")
    for gpu in gpus:
        if "4090D" not in "".join(gpu["name"].split()).upper() or gpu["memory_bytes"] < 23 * 1024 ** 3:
            errors.append("Expected RTX 4090D 24GB")
    # CPU allocation/model are recorded separately: affinity may expose the host's
    # physical CPUs instead of the provider's advertised 18-vCPU allocation.
    if reference is not None:
        for key in ("python", "torch", "torch_cuda_runtime", "cudnn", "packages"):
            if actual.get(key) != reference.get(key):
                errors.append("CASE-EPCL runtime differs: " + key)
    return errors


def collect():
    # Only this explicit cloud-only command imports torch. Other suite tools do not.
    import torch
    actual = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "packages": {}, "gpus": [],
              "cpu_count_reported": os.cpu_count(),
              "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    if sys.platform == "linux":
        actual["os_release"] = dict(platform.freedesktop_os_release())
        actual["cpu_model_lines"] = sorted(set(line.strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                               if line.startswith("model name")))
        quota = Path("/sys/fs/cgroup/cpu.max")
        actual["cgroup_cpu_max"] = quota.read_text().strip() if quota.exists() else None
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            actual["gpus"].append({"name": properties.name, "memory_bytes": properties.total_memory})
    for name in PACKAGES:
        try:
            actual["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual["packages"][name] = None
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=HERE / "cloud_target.json")
    parser.add_argument("--reference", type=Path, help="Snapshot made by this tool in the actual CASE-EPCL cloud environment")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    target = json.loads(args.target.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))["actual"] if args.reference else None
    actual = collect()
    errors = validate(actual, target, reference)
    result = {"target": target, "actual": actual, "errors": errors,
              "target_runtime_passed": not errors, "case_epcl_reference_compared": reference is not None,
              "training_started": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"target_runtime_passed": not errors, "case_epcl_reference_compared": reference is not None}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
