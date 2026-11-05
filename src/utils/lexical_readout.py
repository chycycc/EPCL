import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.special import softmax


class LexicalEmotionReadout:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "expert.json").read_text(encoding="utf-8"))
        assert self.metadata["format"] == "t7_first_utterance_lexical_v1"
        assert self.metadata["classes"] == list(range(32))
        self.vocabulary = self.metadata["vocabulary"]
        with np.load(directory / "expert.npz") as arrays:
            self.idf = arrays["idf"].copy()
            self.coef = arrays["coef"].copy()
            self.intercept = arrays["intercept"].copy()

    def __call__(self, logits, batch):
        rows, cols, values = [], [], []
        for row, context in enumerate(batch["input_txt"]):
            tokens = " ".join(context[0]).lower().split()
            counts = Counter(tokens)
            counts.update(a + " " + b for a, b in zip(tokens, tokens[1:]))
            for term, count in counts.items():
                col = self.vocabulary.get(term)
                if col is not None:
                    rows.append(row)
                    cols.append(col)
                    values.append((1.0 + np.log(count)) * self.idf[col])
        x = csr_matrix((values, (rows, cols)),
                       shape=(len(batch["input_txt"]), len(self.vocabulary)), dtype=np.float64)
        norm = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        x = x.multiply((1.0 / np.maximum(norm, 1e-300))[:, None]).tocsr()
        lexical = x @ self.coef.T + self.intercept
        meta = self.metadata
        base = logits.detach().cpu().numpy().astype(np.float64)
        prob = ((1.0 - meta["alpha"]) * softmax(base / meta["T_base"], axis=1)
                + meta["alpha"] * softmax(lexical / meta["T_lexical"], axis=1))
        return torch.as_tensor(np.log(np.clip(prob, 1e-15, 1.0)),
                               dtype=logits.dtype, device=logits.device)


def export_frozen_expert(source, destination):
    import pickle
    with open(source, "rb") as stream:
        expert = pickle.load(stream)
    vectorizer, classifier = expert["vectorizer"], expert["classifier"]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {key: float(expert[key]) for key in ("alpha", "T_base", "T_lexical")}
    metadata.update(format="t7_first_utterance_lexical_v1",
                    classes=classifier.classes_.tolist(),
                    vocabulary={key: int(value) for key, value in vectorizer.vocabulary_.items()},
                    source="first source utterance; word 1-2 grams; sublinear TF; L2 TFIDF",
                    base="T7 composite7499; masked token pooling")
    (destination / "expert.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(destination / "expert.npz", idf=vectorizer.idf_,
                        coef=classifier.coef_, intercept=classifier.intercept_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_pickle", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    export_frozen_expert(args.export_pickle, args.output_dir)
