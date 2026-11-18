"""Train independently in the cloud; evaluate returned weights locally."""
import argparse
import json
import math
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
LIB = HERE / "lib" if (HERE / "lib").exists() else HERE.parent / "classic_baselines"
sys.path.insert(0, str(LIB))
import torch
from tqdm.auto import tqdm
from train_baseline import fit, ordered_loader, write_json, code_version
from native_runtime import readonly_observation
from score_records import likelihood, diversity, read_jsonl
from training_state import restore_weights
from native_generation import tokenize_response
from runtime import Runtime, same_text
from search import beam

MODELS = ("iamm", "iamm_no_situation", "escm", "seek")


def evaluate(runtime, output, recipe, version, generation_n, max_steps):
    best = torch.load(output / "ppl_best.pt", map_location="cpu")
    assert best["recipe"]["model"] == runtime.name
    restore_weights(runtime, best["weights"])
    runtime.model.eval()
    canonical = read_jsonl(runtime.shared_data / "canonical/test.jsonl")
    dataset = runtime.datasets["test"]
    assert len(canonical) == len(dataset) == 5255
    records, generated = [], []
    with (output / "test_records.jsonl").open("w", encoding="utf-8") as stream, \
            (output / f"generation_{generation_n}.jsonl").open("w", encoding="utf-8") as generations:
        for index, batch in enumerate(tqdm(ordered_loader(runtime, "test", 1), desc="Test")):
            gold = canonical[index]
            # Native extra features stay attached to their matching dialogue.
            assert same_text(dataset.data["target"][index], gold["target"]), f"Target/order mismatch at {index}"
            record = dict(sample_id=gold["sample_id"], emotion_gold=gold["emotion"], **runtime.teacher_record(batch))
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if index < generation_n:
                text = runtime.greedy(batch, max_steps)
                response = dict(sample_id=gold["sample_id"], greedy_text=text, greedy_tokens=tokenize_response(text),
                                **beam(runtime, batch, max_steps))
                generated.append(response)
                generations.write(json.dumps(response, ensure_ascii=False) + "\n")
    metrics = dict(model=runtime.name, checkpoint="ppl_best.pt", checkpoint_main_step=best["progress"]["best_step"],
                   checkpoint_git_commit=best["git_commit"], evaluation_git_commit=version,
                   test=likelihood(records), generation_n=len(generated), max_dec_step=max_steps,
                   test_batch_size=1, shuffle=False, decoding="native_greedy",
                   vocabulary_size=runtime.vocab.n_words, target_encoding="author_native_tokenization_and_vocabulary",
                   situation_input=runtime.name == "iamm", extra_supervision="41-class utterance emotion/intent" if runtime.name == "seek" else None,
                   generation={mode: diversity([r[mode + "_tokens"] for r in generated]) for mode in ("greedy", "beam")}, recipe=recipe)
    metrics["test"]["case_ppl"] = math.exp(math.fsum(r["nll_sum"] / r["token_count"] for r in records) / len(records))
    write_json(output / "eval_metrics.json", metrics)
    print("[Evaluation Complete] " + str(output / "eval_metrics.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--bundle-root", type=Path, default=HERE.parent)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--shared-data", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--embedding-file", type=Path, default=Path("/root/autodl-tmp/CASE/vectors/glove.6B.300d.txt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gen-samples", type=int, default=500)
    parser.add_argument("--max-dec-step", type=int, default=40)
    parser.add_argument("--smoke-updates", type=int, default=0)
    args = parser.parse_args()
    assert not (args.resume and args.evaluate_only)
    assert 1 <= args.gen_samples <= 5255 and args.max_dec_step > 0
    bundle = args.bundle_root.resolve()
    workspaces = (args.workspace_root or bundle / "workspace_extended").resolve()
    shared = (args.shared_data or bundle / "data").resolve()
    output = (args.run_dir or bundle / "runs" / ("formal_" + args.model)).resolve()
    if not (args.resume or args.evaluate_only):
        output.mkdir(parents=True, exist_ok=False)
    recipe = json.loads((HERE / "configs" / (args.model + ".json")).read_text())
    family = "iamm" if args.model.startswith("iamm") else args.model
    torch.set_num_threads(1)
    runtime = Runtime(args.model, workspaces / family, shared, output, args.embedding_file,
                      args.device, seed=recipe["seed"], batch_size=recipe["batch_size"])
    runtime.shared_data = shared
    torch.set_num_threads(1)
    version_file = HERE / "release.json"
    version = json.loads(version_file.read_text())["git_commit"] if version_file.exists() else code_version()
    if args.smoke_updates:
        from training_state import capture, restore_state
        runtime.model.train()
        parameter = runtime.model.embedding.lut.weight
        before = parameter.detach().clone()
        for step in range(args.smoke_updates):
            values = runtime.model.train_one_batch(runtime.batch("train", step), step)
            assert all(math.isfinite(v) for v in values)
        assert not torch.equal(parameter, before), "No embedding update"
        torch.save(capture(runtime, args.smoke_updates), output / "smoke.pt")
        restore_state(runtime, torch.load(output / "smoke.pt", map_location="cpu"))
        runtime.model.eval()
        batch = runtime.batch("test", 0)
        record = runtime.teacher_record(batch)
        assert math.isfinite(record["nll_sum"]) and math.isfinite(record["emotion_ce"])
        from search import Distribution
        from native_generation import search
        with readonly_observation(runtime.model):
            greedy_ids = search(Distribution(runtime, batch), 5, 1)
            content = greedy_ids[:greedy_ids.index(2)] if 2 in greedy_ids else greedy_ids
            common_text = " ".join(runtime.vocab.index2word[i] for i in content)
            assert tokenize_response(common_text) == tokenize_response(runtime.greedy(batch, 5))
            print(beam(runtime, batch, 5))
        print(record)
        print("[Smoke Complete] update, restore, NLL/classifier, native Greedy agreement, Beam", flush=True)
        return
    assert runtime.model.optimizer.warmup == recipe["noam_warmup"]
    if not (args.resume or args.evaluate_only):
        write_json(output / "run.json", dict(recipe=recipe, git_commit=version, source=runtime.provenance,
                   effective_config=runtime.effective_config, environment=dict(python=platform.python_version(),
                   torch=torch.__version__, cuda=torch.version.cuda, device=args.device)))
    if not args.evaluate_only:
        saved = torch.load(output / "last.pt", map_location="cpu") if args.resume else None
        fit(runtime, recipe, output, version, resume=saved)
        print("[Training Complete] Download this run for local evaluation: " + str(output), flush=True)
    else:
        evaluate(runtime, output, recipe, version, args.gen_samples, args.max_dec_step)


if __name__ == "__main__":
    main()
