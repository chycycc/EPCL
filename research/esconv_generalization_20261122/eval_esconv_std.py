#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESConv 跨数据集泛化实验全量统一标准评测脚本（Greedy + Beam-5 + Strategy/Emotion + BLEU/ROUGE-L + 原型有效秩 + 显著性检验）。

评测口径（严格对齐 CASE-EPCL 审计铁律 v3 与 ESConv 文献惯例）：
1. 测试集：ESConv Session 级防泄漏全量 Test 集（N=1817），固定 batch_size=1，shuffle=False；
2. 似然指标：
   - case_ppl = exp((1 / N) * sum_i (NLL_i / L_i))（与 CASE 原版 evaluate 口径一致）
   - token_micro_ppl = exp((sum_i NLL_i) / (sum_i L_i))（语料词元级口径）
3. 判别指标：
   - 8 类支持策略（ESC_STRATEGY_MAP 索引 7..14）：Str CE、Str Acc (%)、Str Macro-F1 (%)
   - 11 类求助者情绪（启用 esconv_enable_emo_epcl 时）：Emo CE、Emo Acc (%)、8 主类 Macro-F1 (%)
4. 生成指标（全量 Greedy 与 OpenNMT Beam-5，max_dec_step=50，NLTK word_tokenize）：
   - 多样性：Dist-1 (%)、Dist-2 (%)、Unique (%)、Avg_Len
   - 相关性（ESConv 常规指标）：BLEU-2 (%)、BLEU-4 (%)、ROUGE-L F1 (%)
5. 原型几何指标（存在原型路径时）：11 类原型 Gram 矩阵有效秩 eff_rank（香农熵指数）与平均成对余弦相似度。
6. 统计显著性（--compare 模式）：跨臂输出 Strategy Acc McNemar 精确检验与逐样本 NLL / Str CE 配对检验。
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from sklearn.metrics import f1_score
from tqdm import tqdm

STRATEGY_CLASS_IDS = list(range(7, 15))  # ESC_STRATEGY_MAP 中 8 类真实支持策略索引


def sha256_file(path):
    """计算检查点或数据文件 SHA256 摘要。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenize_nltk(text):
    """使用 NLTK word_tokenize 对小写规范化文本分词，与 ED 全量标准评测保持一致。"""
    text = (text or "").strip().lower()
    if not text:
        return []
    return nltk.word_tokenize(text)


def lcs_length(a, b):
    """计算两个词序列的最长公共子序列长度（O(min(len)) 空间）。"""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, 1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[-1]


def compute_generation_metrics(hyps, refs):
    """统一计算 Distinct-1/2、Unique、Avg_Len、BLEU-2/4 与 ROUGE-L F1（均为百分制）。"""
    assert len(hyps) == len(refs) and len(hyps) > 0
    hyp_toks_list = [tokenize_nltk(h) for h in hyps]
    ref_toks_list = [tokenize_nltk(r) for r in refs]

    unigrams, bigrams = set(), set()
    total_uni, total_bi = 0, 0
    lens = []
    norm_sents = []

    rouge_l_f1s = []
    for h_toks, r_toks in zip(hyp_toks_list, ref_toks_list):
        lens.append(len(h_toks))
        norm_sents.append(" ".join(h_toks))
        total_uni += len(h_toks)
        for t in h_toks:
            unigrams.add(t)
        if len(h_toks) >= 2:
            total_bi += len(h_toks) - 1
            for i in range(len(h_toks) - 1):
                bigrams.add((h_toks[i], h_toks[i + 1]))

        if len(h_toks) == 0 or len(r_toks) == 0:
            rouge_l_f1s.append(0.0)
        else:
            lcs = lcs_length(h_toks, r_toks)
            prec = lcs / float(len(h_toks))
            rec = lcs / float(len(r_toks))
            f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
            rouge_l_f1s.append(f1)

    smooth = SmoothingFunction().method1
    refs_wrapped = [[r] for r in ref_toks_list]
    bleu2 = corpus_bleu(refs_wrapped, hyp_toks_list, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_wrapped, hyp_toks_list, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "Dist-1": round(100.0 * len(unigrams) / max(1, total_uni), 4),
        "Dist-2": round(100.0 * len(bigrams) / max(1, total_bi), 4),
        "Unique": round(100.0 * len(set(norm_sents)) / float(len(norm_sents)), 4),
        "Avg_Len": round(float(np.mean(lens)), 4),
        "BLEU-2": round(100.0 * float(bleu2), 4),
        "BLEU-4": round(100.0 * float(bleu4), 4),
        "ROUGE-L": round(100.0 * float(np.mean(rouge_l_f1s)), 4),
    }


def compute_prototype_geometry(model):
    """计算原型矩阵的 Gram 有效秩（eff_rank）与非对角平均余弦相似度。"""
    if not hasattr(model, "epcl_criterion") or not hasattr(model.epcl_criterion, "current_prototypes"):
        return None
    if not getattr(model, "emo_epcl_active", False) or getattr(model, "disable_all_prototypes", False):
        return None
    with torch.no_grad():
        protos = model.epcl_criterion.current_prototypes.detach().float().cpu()
        protos = F.normalize(protos, p=2, dim=1)
        gram = torch.matmul(protos, protos.T)
        eigvals = torch.linalg.eigvalsh(gram).clamp(min=0.0)
        eig_sum = eigvals.sum().item()
        if eig_sum <= 0:
            eff_rank = 0.0
        else:
            p = (eigvals / eig_sum).clamp(min=1e-12)
            entropy = -(p * torch.log(p)).sum().item()
            eff_rank = float(math.exp(entropy))
        n = gram.size(0)
        off_diag = gram[~torch.eye(n, dtype=torch.bool)]
        return {
            "num_prototypes": int(n),
            "eff_rank": round(eff_rank, 4),
            "off_diag_cos_mean": round(float(off_diag.mean().item()), 4),
            "off_diag_cos_max": round(float(off_diag.max().item()), 4),
            "off_diag_cos_min": round(float(off_diag.min().item()), 4),
        }


def find_dual_best_checkpoints(save_dir):
    """在训练输出目录中自动定位 composite_best 与 ppl_best 检查点，并检测是否指向同一步。"""
    save_dir = Path(save_dir)
    comp_candidates = sorted(save_dir.glob("CASE_*_composite_best"))
    ppl_candidates = sorted(save_dir.glob("CASE_*_ppl_best"))
    assert len(comp_candidates) == 1, f"期望找到唯一 composite_best 检查点，实际找到 {len(comp_candidates)}: {comp_candidates}"
    assert len(ppl_candidates) == 1, f"期望找到唯一 ppl_best 检查点，实际找到 {len(ppl_candidates)}: {ppl_candidates}"
    return comp_candidates[0], ppl_candidates[0]


def infer_topology_from_checkpoint(ckpt_path):
    """从检查点字典及其 state_dict 严格反推模型拓扑开关，避免评测期配置错配。"""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    has_pcam = any(k.startswith("pcam.") for k in sd.keys())
    has_pcam_inner_ln = bool(ckpt.get("pcam_inner_ln", False)) if isinstance(ckpt, dict) else False
    use_mcp = "epcl_criterion.class_initialized" in sd
    return {
        "iter": int(ckpt.get("iter", -1)) if isinstance(ckpt, dict) else -1,
        "has_pcam": has_pcam,
        "pcam_inner_ln": has_pcam_inner_ln,
        "use_mcp": use_mcp,
        "use_pcmg_head": bool(ckpt.get("use_pcmg_head", False)) if isinstance(ckpt, dict) else False,
        "use_masked_react_pool": bool(ckpt.get("use_masked_react_pool", False)) if isinstance(ckpt, dict) else False,
        "use_fine_mim_valid_neg": bool(ckpt.get("use_fine_mim_valid_neg", False)) if isinstance(ckpt, dict) else False,
        "disable_epcl_loss": bool(ckpt.get("disable_epcl_loss", False)) if isinstance(ckpt, dict) else False,
        "disable_all_prototypes": bool(ckpt.get("disable_all_prototypes", False)) if isinstance(ckpt, dict) else False,
        "esconv_enable_emo_epcl": bool(ckpt.get("esconv_enable_emo_epcl", False)) if isinstance(ckpt, dict) else False,
        "esconv_emo_loss_weight": float(ckpt.get("esconv_emo_loss_weight", 1.0)) if isinstance(ckpt, dict) else 1.0,
    }


def evaluate_single_checkpoint(ckpt_path, role_name, out_dir, data_dir="data", gpu=0, max_dec_step=50, lambda_epcl=0.03):
    """对单个 ESConv 检查点执行全量 Test (batch=1, shuffle=False) 的似然+分类+Greedy+Beam-5 评测。"""
    from src.models.CASE.model import CASE
    from src.utils.common import set_seed
    from src.utils.config import config
    from src.utils.data.loader import prepare_data_seq
    from src.utils.decode.case import Translator

    topo = infer_topology_from_checkpoint(ckpt_path)

    # 严格对齐全局 config（确保与训练检查点逐项一致）
    config.data_dir = data_dir
    config.dataset = "ESConv"
    config.model = "case"
    config.seed = 13
    config.test = True
    config.pretrain = False
    config.woStrategy = False
    config.gpu = gpu
    config.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    config.batch_size = 1
    config.beam_size = 5
    config.lambda_epcl = lambda_epcl
    config.use_mcp = topo["use_mcp"]
    config.mcp_momentum = 0.96
    config.alpha_uni = 0.8
    config.use_arc_margin = True
    config.arc_margin = 0.30
    config.arc_mode = "cos"
    config.epcl_anchor = "fine_emotion"
    config.cls_anchor = "default"
    config.use_pcam = topo["has_pcam"] and not topo["disable_all_prototypes"]
    config.pcam_inner_ln = topo["pcam_inner_ln"]
    config.pcam_heads = 2
    config.pcam_dropout = 0.1
    config.pcam_gate_bias = -1.0
    config.pcam_tau_attn = 1.5
    config.use_pcmg_head = topo["use_pcmg_head"]
    config.use_masked_react_pool = topo["use_masked_react_pool"]
    config.use_fine_mim_valid_neg = topo["use_fine_mim_valid_neg"]
    config.disable_epcl_loss = topo["disable_epcl_loss"]
    config.disable_all_prototypes = topo["disable_all_prototypes"]
    config.esconv_enable_emo_epcl = topo["esconv_enable_emo_epcl"]
    config.esconv_emo_loss_weight = topo["esconv_emo_loss_weight"]
    config.use_pcgrad = False
    config.emo_branch_dropout = 0.0
    config.save_path = str(out_dir)

    set_seed()
    _, _, test_set, vocab, emo_num, strategy_num = prepare_data_seq(batch_size=1)

    model = CASE(
        vocab,
        emotion_num=emo_num,
        strategy_num=strategy_num,
        is_eval=True,
        model_file_path=str(ckpt_path),
    )
    model = model.to(config.device)
    model.eval()
    translator = Translator(model, model.vocab)

    n_samples = len(test_set)
    ctx_losses = []
    token_nll_sum = 0.0
    token_count = 0

    str_ce_list, str_true_list, str_pred_list = [], [], []
    emo_ce_list, emo_true_list, emo_pred_list = [], [], []

    refs, hyps_g, hyps_b = [], [], []
    per_sample_records = []

    t0 = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_set, desc=f"Eval {role_name} (N={n_samples})")):
            bow, kl, mim, ctx, ppl, str_l, str_a, emo_l, emo_a, epcl_l, dec_emo_l = model.train_one_batch(
                batch, 0, train=False
            )
            tgt = batch["target_batch"]
            valid_toks = int(tgt.ne(config.PAD_idx).sum().item())
            assert valid_toks > 0
            ctx_losses.append(float(ctx))
            token_nll_sum += float(ctx) * valid_toks
            token_count += valid_toks

            str_label = int(batch["strategy_label"].item())
            str_pred = int(model.current_pred_strategy[0])
            str_ce_list.append(float(str_l))
            str_true_list.append(str_label)
            str_pred_list.append(str_pred)

            emo_label = int(batch["program_label"].item())
            emo_pred = int(model.current_pred_emotion[0]) if model.current_pred_emotion is not None else -1
            if topo["esconv_enable_emo_epcl"]:
                emo_ce_list.append(float(emo_l))
                emo_true_list.append(emo_label)
                emo_pred_list.append(emo_pred)

            sent_g = model.decoder_greedy(batch, max_dec_step=max_dec_step)[0]
            sent_b = translator.beam_search(batch, max_dec_step=max_dec_step)[0]
            ref = " ".join(batch["target_txt"][0])

            refs.append(ref)
            hyps_g.append(sent_g)
            hyps_b.append(sent_b)

            per_sample_records.append({
                "idx": idx,
                "nll_mean": float(ctx),
                "target_tokens": valid_toks,
                "str_true": str_label,
                "str_pred": str_pred,
                "str_correct": int(str_pred == str_label),
                "str_ce": float(str_l),
                "emo_true": emo_label,
                "emo_pred": emo_pred,
                "emo_correct": int(emo_pred == emo_label) if topo["esconv_enable_emo_epcl"] else None,
                "emo_ce": float(emo_l) if topo["esconv_enable_emo_epcl"] else None,
                "ref": ref,
                "greedy": sent_g,
                "beam": sent_b,
            })

    elapsed = time.time() - t0
    case_ppl = float(math.exp(float(np.mean(ctx_losses))))
    token_micro_ppl = float(math.exp(token_nll_sum / max(1, token_count)))
    str_acc = 100.0 * float(np.mean([int(p == y) for p, y in zip(str_pred_list, str_true_list)]))
    str_macro_f1 = 100.0 * float(
        f1_score(str_true_list, str_pred_list, labels=STRATEGY_CLASS_IDS, average="macro", zero_division=0)
    )
    str_invalid_preds = int(sum(1 for p in str_pred_list if p not in STRATEGY_CLASS_IDS))

    emo_metrics = None
    if topo["esconv_enable_emo_epcl"]:
        present_emo_classes = sorted(set(emo_true_list))
        emo_acc = 100.0 * float(np.mean([int(p == y) for p, y in zip(emo_pred_list, emo_true_list)]))
        emo_macro_f1 = 100.0 * float(
            f1_score(emo_true_list, emo_pred_list, labels=present_emo_classes, average="macro", zero_division=0)
        )
        emo_metrics = {
            "Emo_CE": round(float(np.mean(emo_ce_list)), 4),
            "Emo_Acc": round(emo_acc, 4),
            "Emo_Macro_F1": round(emo_macro_f1, 4),
            "evaluated_classes": present_emo_classes,
        }

    greedy_gen = compute_generation_metrics(hyps_g, refs)
    beam_gen = compute_generation_metrics(hyps_b, refs)
    proto_geom = compute_prototype_geometry(model)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{role_name}_per_sample.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in per_sample_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "role": role_name,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "step": topo["iter"],
        "topology": topo,
        "protocol": {
            "dataset": "ESConv",
            "test_samples": n_samples,
            "test_target_tokens": token_count,
            "vocab_size": int(vocab.n_words),
            "batch_size": 1,
            "shuffle": False,
            "max_dec_step": max_dec_step,
            "beam_size": 5,
            "elapsed_sec": round(elapsed, 2),
        },
        "metrics": {
            "case_ppl": round(case_ppl, 4),
            "token_micro_ppl": round(token_micro_ppl, 4),
            "ctx_loss_mean": round(float(np.mean(ctx_losses)), 5),
            "Str_CE": round(float(np.mean(str_ce_list)), 4),
            "Str_Acc": round(str_acc, 4),
            "Str_Macro_F1": round(str_macro_f1, 4),
            "Str_invalid_special_preds": str_invalid_preds,
            "emotion": emo_metrics,
            "Greedy": greedy_gen,
            "Beam": beam_gen,
            "prototype_geometry": proto_geom,
        },
        "per_sample_jsonl": str(jsonl_path),
    }

    json_path = out_dir / f"{role_name}_eval_std.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Eval Done] {role_name} (Step {topo['iter']}): PPL={case_ppl:.4f}, Str_Acc={str_acc:.2f}%, "
          f"Str_F1={str_macro_f1:.2f}%, GD2={greedy_gen['Dist-2']:.2f}%, BD2={beam_gen['Dist-2']:.2f}%, "
          f"G_B2={greedy_gen['BLEU-2']:.2f}%, G_RL={greedy_gen['ROUGE-L']:.2f}%")
    return summary


def mcnemar_exact_pvalue(correct_a, correct_b):
    """双侧精确二项式 McNemar 检验（零外部 scipy 依赖）。"""
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a == 1 and bb == 0)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if a == 0 and bb == 1)
    n = b + c
    if n == 0:
        return {"b_only_a": b, "c_only_b": c, "p_value": 1.0}
    k = min(b, c)
    # 精确计算 sum_{i=0..k} C(n, i) * 0.5^n，双侧乘 2
    prob = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return {"b_only_a": b, "c_only_b": c, "p_value": min(1.0, 2.0 * prob)}


def paired_t_pvalue(vals_a, vals_b):
    """计算配对均值差与双侧正态近似 t 检验（大样本 N=1817 下与 t_{1816} 精度一致）。"""
    diff = np.asarray(vals_a, dtype=np.float64) - np.asarray(vals_b, dtype=np.float64)
    n = len(diff)
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    if std_d == 0.0:
        return {"mean_diff": mean_d, "t_stat": 0.0, "p_value": 1.0}
    t_stat = mean_d / (std_d / math.sqrt(n))
    p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return {"mean_diff": round(mean_d, 6), "t_stat": round(t_stat, 4), "p_value": p_val}


def compare_arms(arm_dirs, out_path):
    """汇总 3 臂双 Best 结果并计算两两配对统计显著性检验。"""
    arms = {}
    for spec in arm_dirs:
        name, d = spec.split("=", 1)
        d = Path(d)
        comp_j = json.load(open(d / "composite_best_eval_std.json", "r", encoding="utf-8"))
        ppl_j = json.load(open(d / "ppl_best_eval_std.json", "r", encoding="utf-8"))
        arms[name] = {"composite_best": comp_j, "ppl_best": ppl_j}

    def load_samples(json_meta):
        rows = []
        with open(json_meta["per_sample_jsonl"], "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    stats = {}
    arm_names = list(arms.keys())
    for role in ("composite_best", "ppl_best"):
        role_stats = {}
        for i in range(len(arm_names)):
            for j in range(i + 1, len(arm_names)):
                a, b = arm_names[i], arm_names[j]
                ra = load_samples(arms[a][role])
                rb = load_samples(arms[b][role])
                assert len(ra) == len(rb)
                pair_key = f"{a}_vs_{b}"
                role_stats[pair_key] = {
                    "str_acc_mcnemar": mcnemar_exact_pvalue(
                        [r["str_correct"] for r in ra], [r["str_correct"] for r in rb]
                    ),
                    "nll_mean_paired": paired_t_pvalue(
                        [r["nll_mean"] for r in ra], [r["nll_mean"] for r in rb]
                    ),
                    "str_ce_paired": paired_t_pvalue(
                        [r["str_ce"] for r in ra], [r["str_ce"] for r in rb]
                    ),
                }
        stats[role] = role_stats

    report = {"arms": arms, "pairwise_significance": stats}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Compare Done] 已输出三臂汇总与显著性检验: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="ESConv 全量标准评测脚本")
    parser.add_argument("--save_dir", type=str, default="", help="单个训练实验目录（包含 composite_best 与 ppl_best）")
    parser.add_argument("--out_dir", type=str, default="", help="评测产出目录（默认 save_dir/eval_std）")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_dec_step", type=int, default=50)
    parser.add_argument("--lambda_epcl", type=float, default=0.03)
    parser.add_argument("--compare", nargs="*", default=None,
                        help="多臂对比模式，格式: baseline=path1 wo_all_proto=path2 case_epcl=path3")
    parser.add_argument("--compare_out", type=str, default="results/esconv/esconv_3arms_summary.json")
    args = parser.parse_args()

    if args.compare:
        compare_arms(args.compare, args.compare_out)
        return

    assert args.save_dir, "必须指定 --save_dir 或 --compare"
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.save_dir) / "eval_std"
    comp_ckpt, ppl_ckpt = find_dual_best_checkpoints(args.save_dir)

    comp_sha = sha256_file(comp_ckpt)
    ppl_sha = sha256_file(ppl_ckpt)

    comp_res = evaluate_single_checkpoint(
        comp_ckpt, "composite_best", out_dir,
        data_dir=args.data_dir, gpu=args.gpu, max_dec_step=args.max_dec_step, lambda_epcl=args.lambda_epcl,
    )
    if comp_sha == ppl_sha:
        # 若双 Best 指向同一权重，严格复用评测结果并显式登记
        ppl_res = dict(comp_res)
        ppl_res["role"] = "ppl_best"
        ppl_res["checkpoint"] = str(ppl_ckpt)
        ppl_res["reused_from_composite_best"] = True
        with open(out_dir / "ppl_best_eval_std.json", "w", encoding="utf-8") as f:
            json.dump(ppl_res, f, indent=2, ensure_ascii=False)
        print("[Eval Reuse] ppl_best 与 composite_best 权重 SHA256 逐位一致，已复用评测结果。")
    else:
        evaluate_single_checkpoint(
            ppl_ckpt, "ppl_best", out_dir,
            data_dir=args.data_dir, gpu=args.gpu, max_dec_step=args.max_dec_step, lambda_epcl=args.lambda_epcl,
        )


if __name__ == "__main__":
    main()
