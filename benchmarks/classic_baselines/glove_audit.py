"""Verify the cloud GloVe file and every vocabulary-matched initialized row."""
from audit_sources import sha256

CLOUD_GLOVE_SHA256 = "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed"


def verify_file(path):
    actual = sha256(path)
    if actual != CLOUD_GLOVE_SHA256:
        raise ValueError("GloVe SHA256 mismatch: " + actual + "; expected cloud original " + CLOUD_GLOVE_SHA256)
    return actual


def verify_initialized(runtime, path):
    import numpy as np
    import torch
    from training_state import fingerprint
    vocabulary = runtime.kemp_vocab[0] if runtime.name == "kemp" else runtime.vocabulary["word2index"]
    weights = runtime.model.embedding.lut.weight.detach().cpu()
    expected = {}
    lines = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            lines += 1
            parts = line.split()
            if len(parts) == weights.shape[1] + 1 and parts[0] in vocabulary:
                expected[vocabulary[parts[0]]] = np.array([float(v) for v in parts[1:]], dtype=np.float32)
    if not expected:
        raise ValueError("No GloVe vocabulary rows matched")
    indices = sorted(expected)
    matrix = torch.from_numpy(np.stack([expected[i] for i in indices]))
    selected = weights[indices]
    if not torch.equal(selected, matrix):
        raise ValueError("Initialized embedding does not exactly match GloVe float32 rows")
    if not torch.isfinite(weights).all() or not runtime.model.embedding.lut.weight.requires_grad:
        raise ValueError("Embedding is nonfinite or unexpectedly frozen")
    return {"file_lines": lines, "matched_rows": len(indices), "vocabulary_rows": len(vocabulary),
            "native_random_rows": len(vocabulary) - len(indices), "dimension": weights.shape[1],
            "matched_rows_bitwise_equal": True, "matched_rows_max_abs_error": 0.0,
            "embedding_sha256_before_training": fingerprint(weights), "embedding_trainable": True}
