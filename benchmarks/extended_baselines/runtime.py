"""Independent IAMM/ESCM/SEEK native runtime; no CASE-EPCL imports."""
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import torch
from native_runtime import seed_all, readonly_observation


def same_text(tokens, other):
    # Author NLTK versions split punctuation differently (e.g. *just*).
    return "".join(tokens) == "".join(other)


class Runtime:
    def __init__(self, name, workspace, shared_data, output, glove, device, seed=13, batch_size=16,
                 random_embeddings=False):
        self.name, self.device = name, device
        glove = glove.resolve()
        self.output = output.resolve()
        self.workspace = workspace.resolve()
        self.family = "iamm" if name.startswith("iamm") else name
        self.provenance = json.loads((self.workspace / "source.json").read_text())
        self.output.mkdir(parents=True, exist_ok=True)
        assert not any(k == "src" or k.startswith("src.") for k in sys.modules), "One native project per process"
        sys.path.insert(0, str(self.workspace))
        os.chdir(self.workspace)
        asset = self.workspace / "data/NRCDict.json"
        asset.parent.mkdir(exist_ok=True)
        if not asset.exists():
            shutil.copyfile(shared_data / "assets/NRCDict.json", asset)
        assert (self.workspace / "data/ED/dataset_preproc.p").is_file(), "Install the author's processed ED data first"
        sys.argv = ["native", "--save_path", str(self.output / "native_save"), "--seed", str(seed),
                    "--batch_size", str(batch_size), "--emb_file", str(glove.resolve())]
        if self.family != "seek":
            sys.argv += ["--model", self.family, "--device_id", "0", "--pointer_gen"]
            sys.argv += ["--dep_dim", "50"] if self.family == "escm" else ["--word_topk", "5", "--ctx_topk", "15", "--cs_topk", "5"]
        if device == "cuda":
            sys.argv += ["--cuda"]
        config = importlib.import_module("src.utils.config")
        config.device = torch.device(device)
        config.pretrain_emb = not random_embeddings
        self.config = config
        loader = importlib.import_module("src.utils.data.loader")
        from src.utils import constants
        train, valid, test, vocab = loader.load_dataset()
        data = dict(train=train, valid=valid, test=test)
        if name == "iamm_no_situation":
            # Remove both the scenario text and scenario-derived COMET for all splits.
            data = {split: dict(values, situation=[["UNK"] * 15 for _ in values["target"]],
                         comet_sit=[[[["UNK", "UNK"]] for _ in range(5)] for _ in values["target"]])
                    for split, values in data.items()}
        self.collate = loader.collate_fn
        if self.family == "iamm":
            native_collate = self.collate
            def collate(items):
                batch = native_collate(items)
                # Native top-k assumes >=15 positions; native COMET mask reads index 1.
                # PAD keeps batch=1 evaluation valid for short inputs in both variants.
                for key in ("input_batch", "mask_input", "input_ext_batch", "situation_batch", "situation_ext_batch"):
                    value = batch[key]
                    batch[key] = torch.nn.functional.pad(value, (0, max(0, 15 - value.size(-1))), value=1)
                for key, value in list(batch.items()):
                    if key.startswith(("c_", "s_")) and torch.is_tensor(value):
                        batch[key] = torch.nn.functional.pad(value, (0, max(0, 2 - value.size(-1))), value=1)
                return batch
            self.collate = collate
        self.datasets = {}
        if self.family == "escm":
            trees = loader.load_dep_tree()[:3]
            vocab, dep_vocab = loader.load_vocab()
            tree_dict = loader.load_tree_dict()
            for (split, values), tree in zip(data.items(), trees):
                self.datasets[split] = loader.Dataset(values, vocab, tree, dep_vocab, tree_dict)
            from src.models.ESCM.model import ESCM
            factory = lambda: ESCM(vocab, decoder_number=32, dep_tree_vocab=dep_vocab, is_eval=False)
        elif self.family == "iamm":
            self.datasets = {s: loader.Dataset(d, vocab) for s, d in data.items()}
            from src.models.IAMM.model import IAMM
            factory = lambda: IAMM(vocab, decoder_number=32, is_eval=False)
        else:
            self.datasets = {s: loader.Dataset(d, vocab) for s, d in data.items()}
            from src.model.SEEK import SEEK
            factory = lambda: SEEK(vocab, decoder_number=len(constants.EMO_MAP), is_eval=False)
        self.vocab = vocab
        canonical_path = shared_data / "canonical/test.jsonl"
        with canonical_path.open(encoding="utf-8") as stream:
            canonical = [json.loads(line) for line in stream if line.strip()]
        assert len(canonical) == len(self.datasets["test"]) == 5255
        for index, gold in enumerate(canonical):
            assert same_text(data["test"]["target"][index], gold["target"]), f"Target/order mismatch at {index}"
            context = data["test"]["context"][index]
            assert len(context) == len(gold["context"]) and all(same_text(a, b) for a, b in zip(context, gold["context"])), f"Context/order mismatch at {index}"
        self.labels = {v: k for k, v in (constants.DIA_EMO_MAP if self.family == "seek" else constants.EMO_MAP).items()}
        seed_all(seed, device)
        self.model = factory().to(device)
        for key, parameter in self.model.named_parameters():
            if parameter.dim() > 1 and key != "embedding.lut.weight" and config.pretrain_emb:
                torch.nn.init.xavier_uniform_(parameter)
        self.native_train = self.model.train_one_batch
        forward = self.model.forward

        def observed_forward(batch):
            result = forward(batch)
            self.logits = result[6] if self.family == "iamm" else result[2] if self.family == "escm" else result[3][4]
            return result
        self.model.forward = observed_forward

        def train_batch(batch, step, train=True):
            self.model.is_eval = not train if self.family != "seek" else False
            raw = self.native_train(batch, step, train=train)
            if train or self.family == "seek":
                ce = float(raw[0])
            else:
                lengths = batch["target_batch"].ne(1).sum(dim=1).tolist()
                ce = math.fsum(loss * length for loss, length in zip(raw[0], lengths)) / sum(lengths)
            acc = float(raw[3][0]) if self.family == "seek" else float(raw[3])
            if step % 100 == 0:
                with (self.output / "native_returns.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(step=step, train=train, returned=raw[:4]), default=float) + "\n")
            return ce, math.exp(min(ce, 100)), float(raw[2]), acc
        self.model.train_one_batch = train_batch
        self.effective_config = {k: v for k, v in vars(config).items() if isinstance(v, (str, int, float, bool))}
        print("[Architecture Fingerprint] " + json.dumps(dict(model=name, source=self.provenance,
              situation_input=name == "iamm", situation_comet=name == "iamm",
              utterance_supervision_classes=len(constants.EMO_MAP) if self.family == "seek" else None,
              dialogue_classes=32, native_config=self.effective_config)), flush=True)

    def batch(self, split, index):
        return self.collate([self.datasets[split][index]])

    def teacher_record(self, batch):
        captured = {}
        def store(module, args, output):
            captured["log_prob"] = output.log() if self.family == "escm" else output
        handle = self.model.generator.register_forward_hook(store)
        try:
            with readonly_observation(self.model):
                native = self.model.train_one_batch(batch, 0, train=False)
                target = batch["target_batch"]
                nll = -captured["log_prob"].double().gather(-1, target.unsqueeze(-1)).squeeze(-1)[target.ne(1)].sum()
                label = batch["dia_emotion_label"] if self.family == "seek" else batch["program_label"]
                logits = self.logits.detach().double()
                ce = torch.nn.functional.cross_entropy(logits, torch.as_tensor(label, device=logits.device).long())
                return dict(nll_sum=nll.item(), token_count=int(target.ne(1).sum()),
                            emotion_pred=self.labels[int(logits.argmax(-1).item())], emotion_ce=ce.item(),
                            native_mean_ce=native[0])
        finally:
            handle.remove()

    def greedy(self, batch, max_steps):
        with readonly_observation(self.model):
            if self.family == "iamm":
                return self.model.decoder_greedy_batch(batch, max_dec_step=max_steps - 1)[0][0].strip()
            return self.model.decoder_greedy(batch, max_dec_step=max_steps - 1)[0].strip()
