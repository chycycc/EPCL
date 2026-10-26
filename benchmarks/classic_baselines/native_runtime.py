"""Load exactly one isolated upstream project in a dedicated worker process."""
import ast
import contextlib
import copy
import importlib
import json
import os
from pathlib import Path
import random
import sys
import types

from audit_data import CacheReader
from audit_sources import sha256
from build_data_bundle import cem_view, kemp_vocabulary


def seed_all(seed, device):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def readonly_observation(model):
    import numpy as np
    import torch
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    frequency = copy.deepcopy(getattr(model, "word_freq", None))
    criterion_weight = getattr(getattr(model, "criterion", None), "weight", None)
    temporary_distributions = [(module, module.true_dist) for module in model.modules() if hasattr(module, "true_dist")]
    try:
        with torch.no_grad():
            yield
    finally:
        if frequency is not None:
            model.word_freq = frequency
        if hasattr(getattr(model, "criterion", None), "weight"):
            model.criterion.weight = criterion_weight
        for module, value in temporary_distributions:
            module.true_dist = value
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class NativeRuntime:
    def __init__(self, model_name, workspace, data_dir, output, device="cpu", seed=13,
                 random_embeddings=False, embedding_file=None):
        import torch
        self.name = model_name
        self.workspace = workspace.resolve()
        self.device = device
        self.output = output.resolve()
        self.data_dir = data_dir.resolve()
        provenance = json.loads((self.workspace / "_baseline_provenance.json").read_text(encoding="utf-8"))
        for name, expected in provenance["prepared_files_sha256"].items():
            if sha256(self.workspace / name) != expected:
                raise ValueError("Prepared source changed: " + name)
        self.provenance = provenance
        self.manifest = json.loads((self.data_dir / "manifest.json").read_text(encoding="utf-8"))
        if sha256(self.data_dir / "shared_data.pkl") != self.manifest["bundle_sha256"]:
            raise ValueError("Data bundle fingerprint mismatch")
        with (self.data_dir / "shared_data.pkl").open("rb") as stream:
            self.data = CacheReader(stream).load()
        self.vocabulary = self.data["vocabulary"]
        self.labels = {index: name for name, index in self.data["emotion_mapping"].items()}
        # Do not let cwd/PYTHONPATH accidentally resolve CASE-EPCL's production src.
        project = Path(__file__).resolve().parents[2]
        for name, module in list(sys.modules.items()):
            if name == "src" or name.startswith("src."):
                raise RuntimeError("Use a fresh worker process; another src package is already imported")
        sys.path[:] = [str(self.workspace)] + [p for p in sys.path if p and Path(p).resolve() != project]
        os.chdir(self.workspace)
        asset = self.workspace / "data/NRCDict.json"
        asset.parent.mkdir(exist_ok=True)
        expected_asset = self.manifest["asset_sha256"]["NRCDict.json"]
        if asset.exists() and sha256(asset) != expected_asset:
            raise ValueError("Refusing to overwrite another NRC lexicon")
        if not asset.exists():
            asset.write_bytes((self.data_dir / "assets/NRCDict.json").read_bytes())
        self.vocab = types.SimpleNamespace(**self.vocabulary)
        native = provenance["native_model"]
        argv = ["baseline_native", "--model", native, "--batch_size", "16",
                "--save_path", str(self.output / "native_save")]
        if embedding_file:
            argv += ["--emb_file", str(embedding_file.resolve())]
        if model_name == "case_original":
            sys.argv = argv + ["--dataset", "ED", "--woStrategy", "--seed", str(seed)]
            from src.utils.config import config
            from src.models.CASE.model import CASE
            from src.utils.data.loader import Dataset
            config.device = torch.device(device)
            config.pretrain = False
            config.pretrain_emb = not random_embeddings
            seed_all(seed, device)
            self.model = CASE(self.vocab, emotion_num=32, strategy_num=8, is_eval=False)
            self.classifier = self.model.emotion_linear
            self.Dataset, self.collate = Dataset, None
        elif model_name in ("cem_original", "multi_trs"):
            sys.argv = argv + ["--seed", str(seed)] + (["--cuda"] if device.startswith("cuda") else [])
            from src.utils import config
            from src.utils.data.loader import Dataset, collate_fn
            config.device = torch.device(device)
            config.pretrain_emb = not random_embeddings
            seed_all(seed, device)
            if model_name == "cem_original":
                from src.models.CEM.model import CEM
                self.model = CEM(self.vocab, decoder_number=32, is_eval=False)
                self.classifier = self.model.emo_lin
            else:
                from src.models.Transformer.model import Transformer
                self.model = Transformer(self.vocab, decoder_number=32, is_multitask=True, is_eval=False)
                self.classifier = self.model.decoder_key
            self.Dataset, self.collate = Dataset, collate_fn
        elif model_name == "kemp":
            # Parse the upstream function without importing its executable training main.
            import argparse
            import logging
            tree = ast.parse((self.workspace / "main.py").read_text(encoding="utf-8"))
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_params")
            namespace = {"os": os, "torch": torch, "argparse": argparse, "logging": logging, "print_opts": lambda args: None}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream_load_params", "exec"), namespace)
            sys.argv = argv + ["--label_smoothing", "--noam", "--emb_dim", "300", "--hidden_dim", "300",
                               "--hop", "1", "--heads", "2", "--concept_num", "1", "--total_concept_num", "10",
                               "--attn_loss", "--pointer_gen", "--pretrain_emb"]
            config = namespace["load_params"]()
            config.device = torch.device(device)
            config.USE_CUDA = device.startswith("cuda")
            config.use_cuda = device.startswith("cuda")
            config.pretrain_emb = not random_embeddings
            for name, value in self.manifest["kemp"]["special_indices"].items():
                setattr(config, name, value)
            # Upstream's package is literally named 'code'. Isolate its import from
            # Python's already-loaded stdlib code module, then restore the latter.
            standard_code = sys.modules.pop("code", None)
            try:
                KEMP = importlib.import_module("code.KEMP").KEMP
                Dataset = importlib.import_module("code.dataloader").Dataset
            finally:
                if standard_code is not None:
                    sys.modules["code"] = standard_code
            seed_all(seed, device)  # KEMP and common_layer reset to zero at import.
            self.kemp_vocab = kemp_vocabulary(self.vocabulary)
            self.model = KEMP(config, self.kemp_vocab, decoder_number=32, is_eval=False)
            self.classifier = self.model.identify
            self.Dataset, self.collate = Dataset, None
        else:
            raise ValueError("Unsupported model: " + model_name)
        self.config = config
        config.seed = seed
        names = vars(config.args) if hasattr(config, "args") else vars(config)
        self.effective_config = {}
        for name in names:
            value = getattr(config, name, None)
            if value is None or isinstance(value, (str, int, float, bool)):
                self.effective_config[name] = value
        self.effective_config.update(device=device, seed=seed, pretrain_emb=config.pretrain_emb,
                                     shared_vocabulary_size=self.vocabulary["n_words"])
        print("[Baseline Effective] " + json.dumps({"model": model_name, "device": device, "seed": seed,
                                                  "pretrain_emb": config.pretrain_emb,
                                                  "pointer_gen": config.pointer_gen,
                                                  "weight_sharing": config.weight_sharing}), flush=True)
        self.model.to(device)
        optimizer = self.model.optimizer
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        self.effective_config["actual_optimizer"] = type(raw_optimizer).__name__
        self.effective_config["actual_noam_warmup"] = getattr(optimizer, "warmup", None)
        self.effective_config["optimizer_betas"] = list(raw_optimizer.param_groups[0].get("betas", ()))
        self.effective_config["optimizer_eps"] = raw_optimizer.param_groups[0].get("eps")
        # Same fresh-training initializer as the pinned authors' main.py.
        for name, parameter in self.model.named_parameters():
            if parameter.dim() > 1 and name != "embedding.lut.weight" and config.pretrain_emb:
                torch.nn.init.xavier_uniform_(parameter)
        self.model.eval()
        self.datasets = {}
        for split, data in self.data["splits"].items():
            if model_name == "kemp":
                self.datasets[split] = self.Dataset(data, self.kemp_vocab[0], config)
            else:
                view = cem_view(data) if model_name in ("cem_original", "multi_trs") else data
                self.datasets[split] = self.Dataset(view, self.vocab)
        self.imported_native_paths = {name: str(module.__file__) for name, module in sys.modules.items()
                                      if (name.startswith("src.") or name.startswith("code.")) and getattr(module, "__file__", None)}
        for name, path in self.imported_native_paths.items():
            if self.workspace not in Path(path).resolve().parents:
                raise RuntimeError("Foreign native module: " + name + " -> " + path)

    def batch(self, split, index):
        dataset = self.datasets[split]
        collate = dataset.collate_fn if self.collate is None else self.collate
        return collate([dataset[index]])

    def teacher_record(self, batch):
        import torch
        captured = {}
        def store(name):
            def hook(module, args, output):
                captured[name] = output.detach()
            return hook
        hooks = [self.model.generator.register_forward_hook(store("log_prob")),
                 self.classifier.register_forward_hook(store("emotion_logits"))]
        try:
            with readonly_observation(self.model):
                native = self.model.train_one_batch(batch, 0, train=False)
                target = batch["target_batch"]
                log_prob = captured["log_prob"]
                if log_prob.shape[-1] != self.vocabulary["n_words"]:
                    raise ValueError("Unexpected pointer OOV extension / output support")
                mask = target.ne(1)
                nll = -log_prob.double().gather(-1, target.unsqueeze(-1)).squeeze(-1)[mask].sum()
                logits = captured["emotion_logits"]
                label = batch["emotion_label"] if self.name == "kemp" else batch["program_label"]
                label = torch.as_tensor(label, dtype=torch.long, device=logits.device).reshape(-1)
                ce = torch.nn.functional.cross_entropy(logits.double(), label)
                native_ce = float(native[3] if self.name == "case_original" else native[0])
                result = {"nll_sum": nll.item(), "token_count": int(mask.sum()),
                          "emotion_pred": self.labels[int(logits.argmax(-1).item())], "emotion_ce": ce.item(),
                          "native_mean_ce": native_ce,
                          "native_ce_abs_error": abs(native_ce - nll.item() / int(mask.sum()))}
                if not all(__import__("math").isfinite(result[k]) for k in ("nll_sum", "emotion_ce", "native_mean_ce")):
                    raise ValueError("Nonfinite native evaluation")
                if result["native_ce_abs_error"] > 1e-5:
                    raise ValueError("Hook NLL does not match native target CE")
                return result
        finally:
            for hook in hooks:
                hook.remove()

    def load_checkpoint(self, path):
        import torch
        checkpoint = torch.load(path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a checkpoint dictionary")
        protocol = checkpoint.get("baseline_protocol", {})
        if protocol.get("target_encoding_sha256") != self.manifest["target_encoding_sha256"]:
            raise ValueError("Checkpoint lacks matching target-encoding provenance")
        if protocol.get("source_commit") != self.provenance["source_commit"]:
            raise ValueError("Checkpoint source provenance differs")
        if type(protocol.get("trained")) is not bool:
            raise ValueError("Checkpoint must explicitly identify trained vs untrained weights")
        self.checkpoint_protocol = protocol
        weights = checkpoint.get("model", checkpoint.get("models", checkpoint))
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
