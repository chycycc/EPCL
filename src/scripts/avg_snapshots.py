import os
import sys
import re
import json
import itertools
import argparse
import gc
import math
import torch
import torch.nn.functional as F

# 确保在导入 src 之前 sys.path 包含工程根目录
sys.path.insert(0, os.getcwd())

from src.utils.config import config
from src.utils.common import set_seed
from src.utils.data.loader import prepare_data_seq
from src.models.CASE.model import CASE
from src.models.common import evaluate


EMOTION_MODULE_PREFIXES = ("emotion_linear.", "epcl_criterion.", "react_linear.")


def parse_args():
    parser = argparse.ArgumentParser(description="[V10 T7] SWA 快照权重平均与验证集最优窗口选择")
    parser.add_argument("--run_dir", type=str, required=True, help="运行主目录（包含 snapshots/ 子目录）")
    parser.add_argument("--composite_ckpt", type=str, required=True, help="composite_best 检查点路径作为模板")
    parser.add_argument("--out_dir", type=str, required=True, help="平均模型权重与选择元数据输出目录")
    parser.add_argument("--max_valid_batches", type=int, default=0, help="验证集评估批次数（0 为全量，用于冒烟测试）")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小（默认继承 config.batch_size）")
    parser.add_argument("--dump_all", action="store_true", default=False, help="是否保存所有候选窗口的平均权重")
    parser.add_argument("--protect_emotion_modules", action="store_true",
                        help="[V10 T8] 保留composite情绪模块，平均候选须在Valid上优于composite")
    args, unknown = parser.parse_known_args()
    return args


def compute_average_weights(snap_models, ref_model, protected_prefixes=()):
    """
    对多个快照的模型权重字典计算平均权重。
    规则严格保持与 scratch/wavg/make_avg.py 一致：
    1. 键集合必须完全一致；
    2. 浮点张量均匀平均（两项时为 0.5 * (a + b)）；
    3. 非浮点张量继承模板检查点；
    4. 受保护模块原样继承模板；未受保护的 prototypes 平均后逐行 L2 归一化。
    """
    for i, m in enumerate(snap_models):
        assert set(m.keys()) == set(ref_model.keys()), \
            f"快照 {i} 的模型键集合与模板检查点不匹配"

    avg_model = {}

    for k, val_ref in ref_model.items():
        if k.startswith(protected_prefixes):
            avg_model[k] = val_ref.clone() if isinstance(val_ref, torch.Tensor) else val_ref
            continue
        if isinstance(val_ref, torch.Tensor) and torch.is_floating_point(val_ref):
            if len(snap_models) == 2:
                avg_t = 0.5 * (snap_models[0][k] + snap_models[1][k])
            else:
                tensors = [m[k] for m in snap_models]
                avg_t = sum(tensors) / float(len(tensors))
            avg_model[k] = avg_t
        else:
            avg_model[k] = val_ref.clone() if isinstance(val_ref, torch.Tensor) else val_ref

    if "epcl_criterion.prototypes" in avg_model and not "epcl_criterion.prototypes".startswith(protected_prefixes):
        avg_model["epcl_criterion.prototypes"] = F.normalize(avg_model["epcl_criterion.prototypes"], p=2, dim=1)
        norm = torch.norm(avg_model["epcl_criterion.prototypes"], p=2, dim=1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), "原型逐行 L2 归一化校验失败"

    return avg_model


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    composite_ckpt = os.path.abspath(args.composite_ckpt)
    out_dir = os.path.abspath(args.out_dir)
    snap_dir = os.path.join(run_dir, "snapshots")

    os.makedirs(out_dir, exist_ok=True)
    config.save_path = out_dir
    protected_prefixes = EMOTION_MODULE_PREFIXES if args.protect_emotion_modules else ()
    if args.protect_emotion_modules:
        assert config.use_emotion_routing and config.use_pcmg_head
        assert config.use_mcp and config.use_pcam and config.lambda_epcl == 0.03

    print(f"[*] 启动 SWA 快照权重平均选择")
    print(f"    - Run Dir       : {run_dir}")
    print(f"    - Snapshots Dir : {snap_dir}")
    print(f"    - Composite Ckpt: {composite_ckpt}")
    print(f"    - Output Dir    : {out_dir}")
    print(f"    - Max Valid Batches: {args.max_valid_batches} (0=全量)")

    assert os.path.exists(composite_ckpt), f"模板检查点不存在: {composite_ckpt}"

    # 1. 初始化验证集与模型结构（优先加载数据集，确保 unpickle 拥有干净连续堆内存）
    config.test = False
    config.pretrain = False
    config.pretrain_emb = False
    set_seed()

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    bs = getattr(config, "batch_size", 8)
    print(f"[*] 加载验证集 DataLoader (batch_size={bs})...")
    _, dev_set, _, vocab, emo_num, strategy_num = prepare_data_seq(batch_size=bs)

    if args.max_valid_batches > 0:
        eval_data = list(itertools.islice(dev_set, args.max_valid_batches))
        print(f"[*] 已截断验证集为前 {len(eval_data)} 个批次供冒烟评估。")
    else:
        eval_data = dev_set

    print("[*] 实例化 CASE 模型结构...")
    model = CASE(vocab, emotion_num=emo_num, strategy_num=strategy_num, is_eval=True, model_file_path=None)
    model.to(config.device)

    # 2. 加载模板检查点模型字典
    print(f"[*] 加载模板检查点: {composite_ckpt} ...")
    template_state = torch.load(composite_ckpt, map_location="cpu")
    ref_model = template_state["model"]

    C = template_state.get("iter")
    if C is None:
        match = re.search(r"CASE_(\d+)_", os.path.basename(composite_ckpt))
        if match:
            C = int(match.group(1))
        else:
            raise ValueError(f"无法从检查点提取步数 C: {composite_ckpt}")
    C = int(C)
    print(f"[*] 识别 composite_best 步数 C = {C}")

    # 临时释放 optimizer 状态张量以减小评估期间的内存常驻（约 240MB）
    saved_optimizer_state = template_state.pop("optimizer", None)
    gc.collect()

    def snapshot_exists(step):
        path = os.path.join(snap_dir, f"SNAP_{step}")
        if os.path.exists(path):
            return True
        if step == C:
            return True
        return False

    def load_snapshot_model(step):
        path = os.path.join(snap_dir, f"SNAP_{step}")
        if os.path.exists(path):
            s = torch.load(path, map_location="cpu")
            m = s["model"] if "model" in s else s
            del s
            return m
        if step == C:
            return ref_model
        return None

    # 3. 预注册候选窗口
    raw_candidate_windows = [
        (C, C + 500),
        (C, C + 1000),
        (C, C + 500, C + 1000),
        (C - 500, C, C + 500),
        (C, C + 500, C + 1000, C + 1500),
    ]

    evaluated_window_tuples = []
    seen_step_tuples = set()

    for cand in raw_candidate_windows:
        avail = tuple(s for s in cand if snapshot_exists(s))
        if len(avail) >= 2 and avail not in seen_step_tuples:
            seen_step_tuples.add(avail)
            evaluated_window_tuples.append(avail)

    if not evaluated_window_tuples and not args.protect_emotion_modules:
        raise RuntimeError(f"在 {snap_dir} 中未找到包含至少 2 个快照的有效候选窗口！")

    print(f"[*] 共发现 {len(evaluated_window_tuples)} 个有效候选平均窗口:")
    for w in evaluated_window_tuples:
        print(f"    - 窗口步数: {list(w)}")

    window_results = []
    best_score = float("inf")
    best_window = None
    best_avg_model = None

    composite_valid = None
    if args.protect_emotion_modules and evaluated_window_tuples:
        model.load_state_dict(ref_model, strict=True)
        model.eval()
        with torch.no_grad():
            baseline = evaluate(model, eval_data, ty="valid", max_dec_step=50)
        baseline_score = baseline[4] + 5.0 * baseline[7]
        assert math.isfinite(baseline_score), "Composite Valid score must be finite"
        composite_valid = {"valid_ppl": float(baseline[4]),
                           "valid_emo_loss": float(baseline[7]),
                           "valid_emo_acc": float(baseline[8]),
                           "score": float(baseline_score)}

    # 4. 逐窗口评估
    for w_steps in evaluated_window_tuples:
        print(f"\n[Evaluating Window] 正在平均并评估窗口快照 {list(w_steps)} ...")
        snap_models = [load_snapshot_model(s) for s in w_steps]
        avg_model_state = compute_average_weights(snap_models, ref_model, protected_prefixes)
        del snap_models
        gc.collect()

        model.load_state_dict(avg_model_state, strict=True)
        model.eval()

        with torch.no_grad():
            eval_res = evaluate(model, eval_data, ty="valid", max_dec_step=50)
            ppl_val = eval_res[4]
            emo_loss_val = eval_res[7]
            emo_acc_val = eval_res[8]

        score = ppl_val + 5.0 * emo_loss_val
        assert math.isfinite(score), "Average Valid score must be finite"
        print(f"[Window {list(w_steps)}] Valid PPL={ppl_val:.4f}, EMO_loss={emo_loss_val:.4f}, EMO_acc={100.0 * emo_acc_val:.2f}%, Score={score:.4f}")

        win_info = {
            "window": list(w_steps),
            "valid_ppl": round(float(ppl_val), 4),
            "valid_emo_loss": round(float(emo_loss_val), 4),
            "valid_emo_acc": round(float(emo_acc_val), 4),
            "score": round(float(score), 4)
        }
        window_results.append(win_info)

        if args.dump_all:
            dump_name = f"CASE_avg_{'_'.join(map(str, w_steps))}"
            dump_path = os.path.join(out_dir, dump_name)
            dump_dict = dict(template_state)
            dump_dict["model"] = avg_model_state
            dump_dict["tag"] = f"swa_{'_'.join(map(str, w_steps))}"
            if saved_optimizer_state is not None:
                dump_dict["optimizer"] = saved_optimizer_state
            torch.save(dump_dict, dump_path)
            print(f"    [Dump] 已导出候选平均权重: {dump_path}")

        if score < best_score:
            best_score = score
            best_window = w_steps
            best_avg_model = avg_model_state

    # T8主候选在Test之前确定；平分保留composite，无窗口时直接跳过SWA。
    selected_swa = best_window is not None and (
        not args.protect_emotion_modules or best_score < composite_valid["score"])
    selected_role = "swa" if selected_swa else "composite"
    selected_checkpoint = composite_ckpt
    best_ckpt_path = None
    best_ckpt_name = None

    del model, dev_set, eval_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if selected_swa:
        best_steps_str = "_".join(map(str, best_window))
        best_ckpt_name = f"CASE_avg_{best_steps_str}"
        best_ckpt_path = os.path.join(out_dir, best_ckpt_name)
        if not (args.dump_all and os.path.exists(best_ckpt_path)):
            best_ckpt_dict = dict(template_state)
            best_ckpt_dict["model"] = best_avg_model
            best_ckpt_dict["tag"] = f"swa_{best_steps_str}"
            if saved_optimizer_state is not None:
                best_ckpt_dict["optimizer"] = saved_optimizer_state
            torch.save(best_ckpt_dict, best_ckpt_path)
        selected_checkpoint = best_ckpt_path
        print(f"[+] 已保存平均主候选: {best_ckpt_path}")

    selection_metadata = {
        "composite_ckpt": composite_ckpt,
        "composite_step": C,
        "selection_metric": "score = valid_ppl + 5.0 * valid_emo_loss",
        "evaluation_scope": "full_valid" if args.max_valid_batches == 0 else "smoke_subset",
        "max_valid_batches": args.max_valid_batches,
        "protected_prefixes": list(protected_prefixes),
        "protected_step": C if protected_prefixes else None,
        "configuration": {key: getattr(config, key) for key in (
            "dataset", "seed", "batch_size", "lambda_epcl", "use_mcp", "mcp_momentum",
            "use_pcam", "pcam_heads", "pcam_tau_attn", "use_arc_margin", "arc_margin",
            "arc_mode", "epcl_anchor", "cls_anchor", "use_pcmg_head", "use_emotion_routing")},
        "composite_valid": composite_valid,
        "evaluated_windows": window_results,
        "best_window": best_window,
        "best_score": float(best_score) if best_window is not None else None,
        "best_ckpt_path": best_ckpt_path,
        "best_ckpt_name": best_ckpt_name,
        "selected_role": selected_role,
        "selected_checkpoint": selected_checkpoint,
        "selection_reason": ("average_lower_valid_score" if selected_swa else
                             "no_valid_window" if best_window is None else "composite_not_worse"),
    }
    print("[SWA Selection] " + json.dumps(selection_metadata, ensure_ascii=False))

    selection_path = os.path.join(out_dir, "avg_selection.json")
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_metadata, f, indent=2, ensure_ascii=False)
    print(f"[+] 已写入选择元数据至: {selection_path}")


if __name__ == "__main__":
    main()
