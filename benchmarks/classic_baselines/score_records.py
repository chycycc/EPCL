"""Score adapter-exported records without importing any training project.

This is an artifact contract, not a model runner or a checkpoint loader.
Native target-only NLL and token counts must be supplied by audited adapters.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def align(records, canonical):
    if not canonical or len(records) != len(canonical):
        raise ValueError("Records must cover the entire supplied canonical split")
    seen = set()
    for record, gold in zip(records, canonical):
        sample_id = record.get("sample_id")
        if sample_id in seen or sample_id != gold["sample_id"]:
            raise ValueError("Duplicate, missing or reordered sample ID")
        seen.add(sample_id)
        if record.get("emotion_gold") != gold["emotion"]:
            raise ValueError("Gold emotion mismatch: " + sample_id)


def finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a number: " + field)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative value: " + field)
    return float(value)


def likelihood(records):
    if not records:
        raise ValueError("Empty likelihood records")
    nlls, lengths = [], []
    for item in records:
        nlls.append(finite_nonnegative(item["nll_sum"], "nll_sum"))
        length = item["token_count"]
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError("token_count must be a positive integer")
        lengths.append(length)
    # Overflow is an error, never silently capped to make a run look finite.
    result = {"n": len(records), "token_count": sum(lengths),
              "sample_macro_ppl": math.fsum(math.exp(n / t) for n, t in zip(nlls, lengths)) / len(records),
              "token_micro_ppl": math.exp(math.fsum(nlls) / sum(lengths))}
    have_classifier = [item.get("emotion_pred") is not None for item in records]
    if any(have_classifier) and not all(have_classifier):
        raise ValueError("Partial emotion predictions")
    if all(have_classifier):
        for item in records:
            if not isinstance(item["emotion_pred"], str) or not isinstance(item["emotion_gold"], str):
                raise ValueError("Export emotion names, not repository-specific integer IDs")
        result["emotion_correct"] = sum(item["emotion_pred"] == item["emotion_gold"] for item in records)
        result["emotion_acc_micro_pct"] = 100 * result["emotion_correct"] / len(records)
        result["emotion_ce_sample_mean"] = math.fsum(
            finite_nonnegative(item["emotion_ce"], "emotion_ce") for item in records
        ) / len(records)
    else:
        if any(item.get("emotion_ce") is not None for item in records):
            raise ValueError("Emotion CE supplied without classifier predictions")
        result.update(emotion_correct=None, emotion_acc_micro_pct=None, emotion_ce_sample_mean=None)
    return result


def diversity(token_lists):
    if not token_lists:
        raise ValueError("Empty generation records")
    sentences = []
    for value in token_lists:
        if not isinstance(value, list) or not all(isinstance(t, str) and t and not any(c.isspace() for c in t) for t in value):
            raise ValueError("Expected tokens after the declared common text normalization")
        sentences.append(tuple(value))
    unigrams = [token for sentence in sentences for token in sentence]
    bigrams = [pair for sentence in sentences for pair in zip(sentence, sentence[1:])]
    return {"n": len(sentences), "dist_1_pct": 100 * len(set(unigrams)) / len(unigrams) if unigrams else 0.0,
            "dist_2_pct": 100 * len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
            "unique_pct": 100 * len(set(sentences)) / len(sentences),
            "avg_len": len(unigrams) / len(sentences), "empty_n": sum(not s for s in sentences)}


def summarize(records, canonical):
    align(records, canonical)
    result = {"likelihood": likelihood(records), "generation": {}}
    for mode in ("greedy", "beam"):
        values = [item.get(mode + "_tokens") for item in records]
        if any(value is not None for value in values):
            if not all(value is not None for value in values):
                raise ValueError("Partial generation: " + mode)
            result["generation"][mode] = {"full": diversity(values), "first_500": diversity(values[:500])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    required = ("model", "checkpoint_sha256", "source_commit", "environment_sha256", "target_encoding_sha256",
                "canonical_sha256", "nll_definition", "text_normalization", "generation_protocol")
    if any(not metadata.get(key) for key in required):
        parser.error("Missing required provenance metadata")
    actual_sha = hashlib.sha256(args.canonical.read_bytes()).hexdigest()
    if metadata["canonical_sha256"] != actual_sha:
        parser.error("Canonical data fingerprint mismatch")
    if metadata["nll_definition"] != "target_nll_sum_exclude_pad_include_eos_no_auxiliary_no_smoothing":
        parser.error("Unsupported NLL definition")
    result = summarize(read_jsonl(args.records), read_jsonl(args.canonical))
    result["metadata"] = metadata
    result["records_sha256"] = hashlib.sha256(args.records.read_bytes()).hexdigest()
    result["comparison_scope"] = "PPL comparable only within a verified identical target encoding and probability support"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"evaluated_n": result["likelihood"]["n"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
