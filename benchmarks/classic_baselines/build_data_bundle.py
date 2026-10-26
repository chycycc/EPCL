"""Build an isolated shared-data benchmark bundle from a read-only CASE cache.

Architecture-specific knowledge layouts are adapted, not model computations.
This is a standardized-data reproduction, not an author-cache bitwise replay.
"""
import argparse
import collections
import gc
import json
from pathlib import Path
import pickle
import shutil

from audit_data import CacheReader, digest, encoded, emotion_map, sample
from audit_sources import sha256

RELATIONS = ("xIntent", "xNeed", "xWant", "xEffect", "xReact")
RESERVED = ("UNK", "PAD", "EOS", "SOS", "USR", "SYS", "CLS", "KG", "SEP")


def cem_view(data):
    result = {key: data[key] for key in ("context", "target", "emotion", "situation", "emotion_context")}
    result["utt_cs"] = [[item[relation] for relation in RELATIONS] for item in data["utt_cs"]]
    return result


def kemp_vocabulary(vocabulary):
    # Rename two role symbols instead of adding aliases: len(word2index), output
    # dimensions and OOV offsets must remain identical to the shared vocabulary.
    rename = {"USR": "[USR]", "SYS": "[SYS]"}
    return [
        {rename.get(word, word): index for word, index in vocabulary["word2index"].items()},
        {rename.get(word, word): count for word, count in vocabulary["word2count"].items()},
        {str(index): rename.get(word, word) for index, word in vocabulary["index2word"].items()},
        vocabulary["n_words"],
    ]


def validate_features(data, vocabulary):
    counts = collections.Counter()
    words = vocabulary["word2index"]
    for index, context in enumerate(data["context"]):
        if data["utt_cs"][index]["uttr"] != context[-1]:
            raise ValueError("Commonsense source utterance is not the final ED context turn: " + str(index))
        for relation in RELATIONS:
            if relation not in data["utt_cs"][index]:
                raise ValueError("Missing commonsense relation: " + relation)
        turns = data["concepts"][index]
        vads = data["vads"][index]
        vad = data["vad"][index]
        if len(context) != len(turns) or len(context) != len(vads) or len(context) != len(vad):
            raise ValueError("Misaligned knowledge turns: " + str(index))
        for utterance, concepts, vectors, intensity in zip(context, turns, vads, vad):
            if len(concepts) != 3 or any(len(part) != len(utterance) for part in concepts):
                raise ValueError("Misaligned concept tokens: " + str(index))
            if len(vectors) != len(utterance) or len(intensity) != len(utterance):
                raise ValueError("Misaligned VAD tokens: " + str(index))
            counts["context_tokens"] += len(utterance)
            counts["context_oov"] += sum(token not in words for token in utterance)
            for concept_words, concept_vads, concept_vad in zip(*concepts):
                if len(concept_words) != len(concept_vads) or len(concept_words) != len(concept_vad):
                    raise ValueError("Misaligned concept VAD values")
                counts["concept_tokens"] += len(concept_words)
                counts["concept_oov"] += sum(token not in words for token in concept_words)
        counts["target_tokens_excluding_eos"] += len(data["target"][index])
        counts["target_oov"] += sum(token not in words for token in data["target"][index])
    if any(counts[name] for name in ("context_oov", "concept_oov", "target_oov")):
        raise ValueError("Shared-support contract failed: native pointer extension would differ")
    return dict(counts)


def build(case_root, output):
    if output.exists():
        raise ValueError("Use a fresh output directory; original caches are never overwritten")
    cache = case_root / "data/ED/dataset_preproc.p"
    before = sha256(cache)
    with cache.open("rb") as stream:
        source = CacheReader(stream).load()
    vocab = vars(source[3])
    if [vocab["index2word"][i] for i in range(9)] != list(RESERVED):
        raise ValueError("Unexpected shared special-token IDs")
    if len(vocab["word2index"]) != vocab["n_words"] or set(vocab["word2index"].values()) != set(range(vocab["n_words"])):
        raise ValueError("Vocabulary IDs must be dense and bijective")
    mapping = emotion_map(case_root / "src/utils/constants.py", "ED_EMO_MAP")
    output.mkdir(parents=True)
    (output / "canonical").mkdir()
    (output / "assets").mkdir()
    shutil.copyfile(case_root / "data/NRCDict.json", output / "assets/NRCDict.json")
    report = {"schema_version": 2, "source_case_cache_sha256": before,
              "protocol": "shared_case_ed_tokens_vocabulary_and_cached_knowledge_v1",
              "author_native_cache_replay": False, "training_started": False,
              "vocabulary_size": vocab["n_words"], "emotion_mapping": mapping,
              "word_to_index_sha256": digest(vocab["word2index"]),
              "target_encoding_sha256": digest({"symbols": [vocab["index2word"][i] for i in range(vocab["n_words"])],
                                                 "pad": 1, "eos": 2, "unk": 0,
                                                 "tokenizer": "frozen_preexisting_CASE_ED_cache_tokens"}),
              "splits": {}}
    payload = {"vocabulary": vocab, "emotion_mapping": mapping, "splits": {}}
    for split, data in zip(("train", "valid", "test"), source[:3]):
        stats = validate_features(data, vocab)
        canonical_file = output / "canonical" / (split + ".jsonl")
        with canonical_file.open("wb") as stream:
            for index in range(len(data["target"])):
                item = sample(data, index)
                content = digest(item)
                stream.write(encoded({"sample_id": "%s:%06d:%s" % (split, index, content[:16]),
                                      "content_sha256": content, "index": index, **item}) + b"\n")
        view = cem_view(data)
        # The three native adapters consume the same core arrays by reference.
        for index in range(len(data["target"])):
            if sample(data, index) != sample(view, index):
                raise ValueError("CEM layout adapter changed a core sample")
        payload["splits"][split] = data
        report["splits"][split] = {"n": len(data["target"]), "canonical_sha256": sha256(canonical_file),
                                   "case_cem_multi_trs_kemp_core_identity_equal": True,
                                   "feature_alignment_counts": stats}
    native_kemp = kemp_vocabulary(vocab)
    if len(native_kemp[0]) != vocab["n_words"] or native_kemp[0]["[USR]"] != 4 or native_kemp[0]["[SYS]"] != 5:
        raise ValueError("KEMP role rename changed output support")
    report["kemp"] = {"knowledge_source": "CASE ED preexisting ConceptNet/VAD cache, not downloaded author KEMP cache",
                      "special_indices": {"UNK_idx": 0, "PAD_idx": 1, "EOS_idx": 2, "SOS_idx": 3,
                                          "USR_idx": 4, "SYS_idx": 5, "CLS_idx": 6, "KG_idx": 7, "SEP_idx": 8},
                      "role_renames": {"USR": "[USR]", "SYS": "[SYS]"},
                      "extra_vocab_keys_added": 0, "native_pointer_extension_oov_count": 0}
    payload["target_encoding_sha256"] = report["target_encoding_sha256"]
    with (output / "shared_data.pkl").open("wb") as stream:
        pickle.dump(payload, stream, protocol=4)
    report["bundle_sha256"] = sha256(output / "shared_data.pkl")
    report["asset_sha256"] = {"NRCDict.json": sha256(output / "assets/NRCDict.json")}
    if sha256(cache) != before:
        raise ValueError("Source cache changed during export")
    (output / "manifest.json").write_bytes(encoded(report) + b"\n")
    del payload, source
    gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.case_root.resolve(), args.output_dir.resolve())
    print(json.dumps({"vocabulary_size": report["vocabulary_size"], "splits": {s: r["n"] for s, r in report["splits"].items()},
                      "bundle_sha256": report["bundle_sha256"]}))


if __name__ == "__main__":
    main()
