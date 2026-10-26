"""Export plain ED sample identities; never import model/data-loader modules."""
import argparse
import ast
import builtins
import collections
import gc
import hashlib
import json
from pathlib import Path
import pickle

from audit_sources import sha256


class VocabularySnapshot:
    """Passive stand-in for the trusted local cache's Lang object."""


class CacheReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("src.utils.data.loader", "Lang"):
            return VocabularySnapshot
        if module == "collections" and name == "defaultdict":
            return collections.defaultdict
        if module == "builtins" and name in ("set", "frozenset", "slice", "int", "float", "list", "dict"):
            return getattr(builtins, name)
        if module in ("numpy", "numpy.core.multiarray") and name in ("ndarray", "dtype", "_reconstruct", "scalar"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError("Unsupported cache object: " + module + "." + name)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def emotion_map(path, variable):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if len(value) != 32 or set(value.values()) != set(range(32)):
                raise ValueError("Expected a bijective 32-class emotion map")
            return value
    raise ValueError("Emotion map not found: " + variable)


def tokens(value):
    if not isinstance(value, (tuple, list)) or not all(isinstance(item, str) for item in value):
        raise ValueError("Expected a list of token strings")
    return list(value)


def sample(data, index):
    # Deliberately exclude model-specific knowledge features and native integer IDs.
    return {"context": [tokens(turn) for turn in data["context"][index]],
            "target": tokens(data["target"][index]), "emotion": data["emotion"][index]}


def inspect_cache(cache, mapping, export_dir=None, canonical_dir=None):
    with cache.open("rb") as stream:
        payload = CacheReader(stream).load()
    if not isinstance(payload, (list, tuple)) or len(payload) != 4:
        raise ValueError("Expected [train, valid, test, vocabulary]")
    vocab = payload[3]
    result = {"cache_sha256": sha256(cache), "vocabulary_size": vocab.n_words,
              "word_to_index_sha256": digest(vocab.word2index),
              "lexical_support_sha256": digest(sorted(vocab.word2index)), "splits": {}}
    for split, data in zip(("train", "valid", "test"), payload[:3]):
        count = len(data["target"])
        if any(len(data[key]) != count for key in ("context", "emotion")):
            raise ValueError("Mismatched parallel arrays: " + split)
        ordered = hashlib.sha256()
        labels = collections.Counter()
        oov = total = 0
        differences = {key: 0 for key in ("context", "target", "emotion")}
        first_differences = []
        reference = (canonical_dir / (split + ".jsonl")).open(encoding="utf-8") if canonical_dir else None
        export = None
        if export_dir:
            export = (export_dir / (split + ".jsonl")).open("wb")
        try:
            for index in range(count):
                item = sample(data, index)
                if item["emotion"] not in mapping:
                    raise ValueError("Unknown emotion: " + str(item["emotion"]))
                content = digest(item)
                record = {"sample_id": "%s:%06d:%s" % (split, index, content[:16]),
                          "content_sha256": content, "index": index, **item}
                if reference:
                    line = reference.readline()
                    if not line:
                        raise ValueError("Canonical split ended early")
                    expected = json.loads(line)
                    changed = [key for key in differences if item[key] != expected[key]]
                    for key in changed:
                        differences[key] += 1
                    if changed and len(first_differences) < 10:
                        first_differences.append({"index": index, "fields": changed})
                ordered.update(encoded(record) + b"\n")
                labels[item["emotion"]] += 1
                oov += sum(token not in vocab.word2index for token in item["target"])
                total += len(item["target"])
                if export:
                    export.write(encoded(record) + b"\n")
            if reference and reference.readline():
                raise ValueError("Canonical split contains extra rows")
        finally:
            if export:
                export.close()
            if reference:
                reference.close()
        result["splits"][split] = {"n": count, "ordered_samples_sha256": ordered.hexdigest(),
                                   "emotion_counts": dict(sorted(labels.items())),
                                   "target_lexical_tokens": total, "native_target_oov_tokens": oov}
        if export_dir:
            result["splits"][split]["jsonl_sha256"] = sha256(export_dir / (split + ".jsonl"))
        if canonical_dir:
            result["splits"][split]["canonical_differences"] = differences
            result["splits"][split]["first_differing_indices"] = first_differences
    del payload
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--cem-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory; existing evidence is never overwritten")
    args.output_dir.mkdir(parents=True)
    mapping = emotion_map(args.case_root / "src/utils/constants.py", "ED_EMO_MAP")
    cem_mapping = emotion_map(args.cem_root / "src/utils/constants.py", "EMO_MAP")
    export = args.output_dir / "canonical"
    export.mkdir()
    case = inspect_cache(args.case_root / "data/ED/dataset_preproc.p", mapping, export)
    cem = inspect_cache(args.cem_root / "data/ED/dataset_preproc.p", cem_mapping, canonical_dir=export)
    raw = {}
    for split in ("train", "dev", "test"):
        for field in ("dialog", "target", "emotion", "situation"):
            filename = "sys_%s_texts.%s.npy" % (field, split)
            paths = [root / "data/ED" / filename for root in (args.case_root, args.cem_root)]
            raw[filename] = {"case_sha256": sha256(paths[0]), "cem_sha256": sha256(paths[1])}
            raw[filename]["same_bytes"] = raw[filename]["case_sha256"] == raw[filename]["cem_sha256"]
    same_samples = all(case["splits"][s]["ordered_samples_sha256"] == cem["splits"][s]["ordered_samples_sha256"]
                       for s in ("train", "valid", "test"))
    result = {"schema_version": 1, "case": case, "local_cem": cem,
              "raw_files": raw, "same_ordered_samples": same_samples,
              "same_emotion_mapping": mapping == cem_mapping, "emotion_mapping": mapping,
              "same_native_vocabulary": case["word_to_index_sha256"] == cem["word_to_index_sha256"],
              "native_ppl_directly_rankable": False,
              "kemp_dataset_audited": False, "training_started": False}
    (args.output_dir / "data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("same_ordered_samples", "same_emotion_mapping", "same_native_vocabulary")}))
    return 0 if same_samples and mapping == cem_mapping else 1


if __name__ == "__main__":
    raise SystemExit(main())
