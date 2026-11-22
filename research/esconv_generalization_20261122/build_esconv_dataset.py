#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESConv 跨数据集泛化实验数据构建脚本（Session 级防泄漏切分 + 批量 COMET-ATOMIC-2020 + ConceptNet VAD 图）。

核心职责：
1. 从官方原始 `data/ESConv/ESConv.json` 出发，执行严格 Session 级分层 8:1:1 切分（1,040 / 130 / 130 对话），
   彻底修复原版 `src/utils/data/loader.py:259-265` 的话轮级随机切分泄漏隐患，并将极少数单会话情绪（<3 会话）保留在训练集，
   保证 11 类求助者情绪在训练集中全部覆盖且验证/测试集无未见类别。
2. 对全部唯一 `(input_event, relation)` 查询去重后使用 `mismayil/comet-bart-ai2` 进行 GPU 批量束搜索（beam=5, return=5），
   支持 JSONL 断点续传，生成与 `src/utils/data/loader.py` 完全同构的 `dataset_comet_preproc.p`。
3. 流式解析 `conceptnet-assertions-5.7.0.csv.gz` 构建 ESConv 专属 `ConceptNet.json` 与 `ConceptNet_VAD_dict.json`，
   执行 `read_concept_vad` 产出最终 `data/ESConv/dataset_preproc.p` 及完整审计报告 `dataset_audit.json`。
"""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy

# 确保仓库根目录在 sys.path 中，使 pickle 序列化严格复用 src.utils.data.loader.Lang
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import nltk
import torch
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from src.utils.constants import ESC_EMO_MAP, ESC_STRATEGY_MAP, WORD_PAIRS

RELATIONS = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]


def sha256_file(path):
    """计算文件 SHA256 摘要，用于数据版本追溯。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(text):
    """规范化空白字符，与原版 CASE loader 保持一致。"""
    return " ".join(text.strip().split())


def process_sent(sentence):
    """小写化、缩写词替换与 NLTK 分词，与原版 CASE loader 保持一致。"""
    sentence = sentence.lower()
    for k, v in WORD_PAIRS.items():
        sentence = sentence.replace(k, v)
    return nltk.word_tokenize(sentence)


def get_wordnet_pos(tag):
    """将 NLTK 词性标签映射至 WordNet 词性。"""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return None


def stratified_session_split(raw_sessions, seed=13):
    """
    执行严格的 Session 级分层 8:1:1 切分（Train=1040, Dev=130, Test=130）。

    规则：
    - 对会话数 < 3 的极少数情绪类别（pain/jealousy/guilt，各 1 条会话），固定归入训练集，
      确保 11 类情绪原型在训练集均能获得更新，且 Dev/Test 不出现零样本类别。
    - 对其余情绪类别按比例分配 Dev/Test 会话数，并通过最大余数法精确对齐总数 Dev=130、Test=130。
    - 校验 Train/Dev/Test 会话索引零交集。
    """
    n_total = len(raw_sessions)
    target_dev = int(0.1 * n_total)
    target_test = int(0.1 * n_total)
    target_train = n_total - target_dev - target_test

    by_emo = defaultdict(list)
    for idx, sess in enumerate(raw_sessions):
        emo = sess["emotion_type"]
        assert emo in ESC_EMO_MAP, f"未知情绪类别: {emo}"
        by_emo[emo].append(idx)

    rng = random.Random(seed)
    for emo in sorted(by_emo.keys()):
        rng.shuffle(by_emo[emo])

    # 区分可分层大类（>=3 会话）与极少数类（<3 会话）
    eligible = {emo: idxs for emo, idxs in by_emo.items() if len(idxs) >= 3}
    rare = {emo: idxs for emo, idxs in by_emo.items() if len(idxs) < 3}
    eligible_total = sum(len(v) for v in eligible.values())

    def allocate_counts(total_slots):
        """最大余数法分配各情绪类别的验证/测试会话配额，保证每类至少 1 条且不超过地板上限。"""
        raw_quota = {emo: total_slots * len(idxs) / float(eligible_total) for emo, idxs in eligible.items()}
        alloc = {emo: max(1, int(q)) for emo, q in raw_quota.items()}
        diff = total_slots - sum(alloc.values())
        if diff > 0:
            order = sorted(eligible.keys(), key=lambda e: (raw_quota[e] - int(raw_quota[e]), len(eligible[e]), e), reverse=True)
            for emo in order[:diff]:
                alloc[emo] += 1
        elif diff < 0:
            order = sorted(eligible.keys(), key=lambda e: (raw_quota[e] - int(raw_quota[e]), len(eligible[e]), e))
            for emo in order:
                if diff == 0:
                    break
                if alloc[emo] > 1:
                    alloc[emo] -= 1
                    diff += 1
        assert sum(alloc.values()) == total_slots, f"配额分配不守恒: {sum(alloc.values())} != {total_slots}"
        return alloc

    dev_quota = allocate_counts(target_dev)
    test_quota = allocate_counts(target_test)

    train_sess_ids, dev_sess_ids, test_sess_ids = [], [], []
    for emo in sorted(rare.keys()):
        train_sess_ids.extend(rare[emo])

    for emo in sorted(eligible.keys()):
        idxs = eligible[emo]
        nd = dev_quota[emo]
        nt = test_quota[emo]
        assert nd + nt < len(idxs), f"类别 {emo} 样本数不足以保留训练集会话: {len(idxs)} <= {nd}+{nt}"
        dev_sess_ids.extend(idxs[:nd])
        test_sess_ids.extend(idxs[nd : nd + nt])
        train_sess_ids.extend(idxs[nd + nt :])

    rng.shuffle(train_sess_ids)
    rng.shuffle(dev_sess_ids)
    rng.shuffle(test_sess_ids)

    set_tr, set_dv, set_te = set(train_sess_ids), set(dev_sess_ids), set(test_sess_ids)
    assert len(set_tr & set_dv) == 0 and len(set_tr & set_te) == 0 and len(set_dv & set_te) == 0, "发现跨划分会话泄漏！"
    assert len(train_sess_ids) == target_train and len(dev_sess_ids) == target_dev and len(test_sess_ids) == target_test
    return train_sess_ids, dev_sess_ids, test_sess_ids


def expand_sessions_to_turns(raw_sessions, sess_ids, max_num_dialog=9):
    """按 CASE 原版上下文窗口（max_num_dialog=9，且窗口内须含 seeker 发言）将会话展开为话轮样本。"""
    turns = []
    for sid in sess_ids:
        session = raw_sessions[sid]
        experience_type = session["experience_type"]
        emotion_type = session["emotion_type"]
        problem_type = session["problem_type"]
        situation = session["situation"]
        dialog = session["dialog"]

        context = []
        context_role = []
        context_strategy_seqs = []
        for info in dialog:
            speaker = info["speaker"]
            if speaker == "seeker":
                context.append(info["content"])
                context_role.append(speaker)
                context_strategy_seqs.append("none")
            elif speaker == "supporter":
                assert "strategy" in info["annotation"], f"Session {sid} 缺少 strategy 标注"
                strategy = info["annotation"]["strategy"]
                assert strategy in ESC_STRATEGY_MAP, f"未知策略标签: {strategy}"
                target = info["content"]
                if "seeker" in context_role[-max_num_dialog:]:
                    turns.append({
                        "session_id": sid,
                        "experience_type": experience_type,
                        "emotion_type": emotion_type,
                        "problem_type": problem_type,
                        "situation": situation,
                        "strategy": strategy,
                        "context_strategy_seqs": deepcopy(context_strategy_seqs[-max_num_dialog:]),
                        "context_role": deepcopy(context_role[-max_num_dialog:]),
                        "context": deepcopy(context[-max_num_dialog:]),
                        "target": target,
                    })
                context.append(target)
                context_role.append(speaker)
                context_strategy_seqs.append(strategy)
            else:
                raise ValueError(f"非法 speaker 字段: {speaker}")
    return turns


def build_or_load_splits(esconv_dir, seed=13, max_num_dialog=9, overwrite_splits=False):
    """生成或加载 Session 级防泄漏的 train/dev/test_data.json，并落盘 split_audit.json。"""
    raw_path = os.path.join(esconv_dir, "ESConv.json")
    assert os.path.isfile(raw_path), f"未找到原始文件: {raw_path}"

    train_file = os.path.join(esconv_dir, "train_data.json")
    dev_file = os.path.join(esconv_dir, "dev_data.json")
    test_file = os.path.join(esconv_dir, "test_data.json")
    audit_file = os.path.join(esconv_dir, "split_audit.json")

    if not overwrite_splits and all(os.path.isfile(p) for p in (train_file, dev_file, test_file, audit_file)):
        print("[Split] 检测到已存在的 Session 级划分文件，直接加载并复验零泄漏。")
        train_turns = json.load(open(train_file, "r", encoding="utf-8"))
        dev_turns = json.load(open(dev_file, "r", encoding="utf-8"))
        test_turns = json.load(open(test_file, "r", encoding="utf-8"))
        s_tr = {t["session_id"] for t in train_turns}
        s_dv = {t["session_id"] for t in dev_turns}
        s_te = {t["session_id"] for t in test_turns}
        assert len(s_tr & s_dv) == 0 and len(s_tr & s_te) == 0 and len(s_dv & s_te) == 0, "已有划分存在会话泄漏！"
        return train_turns, dev_turns, test_turns

    raw_sessions = json.load(open(raw_path, "r", encoding="utf-8"))
    tr_sids, dv_sids, te_sids = stratified_session_split(raw_sessions, seed=seed)

    train_turns = expand_sessions_to_turns(raw_sessions, tr_sids, max_num_dialog=max_num_dialog)
    dev_turns = expand_sessions_to_turns(raw_sessions, dv_sids, max_num_dialog=max_num_dialog)
    test_turns = expand_sessions_to_turns(raw_sessions, te_sids, max_num_dialog=max_num_dialog)

    for path, payload in ((train_file, train_turns), (dev_file, dev_turns), (test_file, test_turns)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def summarize(turns, sids):
        return {
            "num_sessions": len(sids),
            "num_turns": len(turns),
            "session_emotion_counts": dict(Counter(raw_sessions[i]["emotion_type"] for i in sids)),
            "turn_emotion_counts": dict(Counter(t["emotion_type"] for t in turns)),
            "turn_strategy_counts": dict(Counter(t["strategy"] for t in turns)),
        }

    # 检查跨划分是否存在完全相同文本的 situation（仅用于审计透明度记录）
    sit_tr = {_norm(raw_sessions[i]["situation"]).lower() for i in tr_sids}
    sit_dv = {_norm(raw_sessions[i]["situation"]).lower() for i in dv_sids}
    sit_te = {_norm(raw_sessions[i]["situation"]).lower() for i in te_sids}

    audit = {
        "split_protocol": "stratified_session_level_8_1_1",
        "seed": seed,
        "max_num_dialog": max_num_dialog,
        "raw_esconv_sha256": sha256_file(raw_path),
        "session_overlap": {
            "train_dev": len(set(tr_sids) & set(dv_sids)),
            "train_test": len(set(tr_sids) & set(te_sids)),
            "dev_test": len(set(dv_sids) & set(te_sids)),
        },
        "exact_situation_text_overlap": {
            "train_dev": len(sit_tr & sit_dv),
            "train_test": len(sit_tr & sit_te),
            "dev_test": len(sit_dv & sit_te),
        },
        "splits": {
            "train": summarize(train_turns, tr_sids),
            "dev": summarize(dev_turns, dv_sids),
            "test": summarize(test_turns, te_sids),
        },
        "session_ids": {
            "train": tr_sids,
            "dev": dv_sids,
            "test": te_sids,
        },
    }
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(
        f"[Split] 完成严格 Session 级分层 8:1:1 切分: "
        f"Train={len(tr_sids)} sess / {len(train_turns)} turns, "
        f"Dev={len(dv_sids)} sess / {len(dev_turns)} turns, "
        f"Test={len(te_sids)} sess / {len(test_turns)} turns | 跨集会话重叠=0"
    )
    return train_turns, dev_turns, test_turns


def pretokenize_split(turns, vocab, stop_words, emotion_lexicon):
    """预分词、构建词表并提取各话轮的唯一 COMET 查询事件，严格对齐 loader.py:172-198。"""
    records = []
    unique_events = set()
    for line in tqdm(turns, desc="Tokenize ESConv"):
        situation = process_sent(_norm(line["situation"]))
        vocab.index_words(situation)
        target = process_sent(_norm(line["target"]))
        vocab.index_words(target)
        strategy = line["strategy"]
        vocab.index_words(process_sent(_norm(strategy)))

        context_list = []
        emotion_list = []
        for uttr in line["context"]:
            item = process_sent(_norm(uttr))
            context_list.append(item)
            vocab.index_words(item)
            ws_pos = nltk.pos_tag(item)
            for w, pos_tag in ws_pos:
                w_p = get_wordnet_pos(pos_tag)
                if w not in stop_words and (w_p == wordnet.ADJ or w in emotion_lexicon) and w not in emotion_list:
                    emotion_list.append(w)

        context_role = line["context_role"]
        last_seeker_idx = max(i for i, r in enumerate(context_role) if r == "seeker")
        last_uttr = context_list[last_seeker_idx]
        last_uttr_str = " ".join(last_uttr)
        split_events = [
            process_sent(_norm(ev))
            for ev in last_uttr_str.split(".")
            if len(process_sent(_norm(ev))) > 0
        ]
        # 兜底：若话语全为句点导致按句点切分为空，则回退为整句，避免 collate 崩溃
        if len(split_events) == 0:
            split_events = [last_uttr]

        unique_events.add(last_uttr_str)
        for ev in split_events:
            unique_events.add(" ".join(ev))
        unique_events.add(" ".join(situation))

        records.append({
            "context_role": context_role,
            "context_strategy_seqs": line["context_strategy_seqs"],
            "emotion": line["emotion_type"],
            "situation": situation,
            "target": target,
            "strategy": strategy,
            "context": context_list,
            "emotion_context": emotion_list,
            "last_uttr": last_uttr,
            "split_events": split_events,
        })
    return records, unique_events


def run_batched_comet(unique_events, comet_model_path, cache_jsonl_path, device, batch_size=128, fp16=True):
    """
    对所有去重后的 (event, rel) 执行批量 COMET-ATOMIC-2020 (BART) 生成，并支持 JSONL 断点续传。

    解码参数与 `src/utils/comet.py` 保持一致：
    - prompt 模板：`"{event} {rel} [GEN]"`
    - `num_beams=5, num_return_sequences=5`
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cache = {}
    if os.path.isfile(cache_jsonl_path):
        with open(cache_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cache[(obj["event"], obj["rel"])] = obj["outputs"]
        print(f"[COMET] 已从断点缓存恢复 {len(cache)} 条查询。")

    pending = []
    for ev in sorted(unique_events):
        for rel in RELATIONS:
            if (ev, rel) not in cache:
                pending.append((ev, rel))

    print(f"[COMET] 唯一事件={len(unique_events)}, 总查询={len(unique_events)*len(RELATIONS)}, 待推理={len(pending)}")
    if not pending:
        return cache

    print(f"[COMET] 正在加载模型: {comet_model_path} -> device={device}, fp16={fp16}")
    tokenizer = AutoTokenizer.from_pretrained(comet_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(comet_model_path)
    if getattr(model.config, "task_specific_params", None):
        pars = model.config.task_specific_params.get("summarization", {})
        model.config.update(pars)
    if fp16 and str(device).startswith("cuda"):
        model = model.half()
    model = model.to(device)
    model.eval()

    # 按输入文本长度排序可显著减少 batch 内 padding，提升 2~3x 吞吐
    pending.sort(key=lambda pair: len(pair[0]))

    t0 = time.time()
    os.makedirs(os.path.dirname(cache_jsonl_path), exist_ok=True)
    with open(cache_jsonl_path, "a", encoding="utf-8") as out_f:
        for start in tqdm(range(0, len(pending), batch_size), desc="COMET Batched Gen"):
            chunk = pending[start : start + batch_size]
            queries = [f"{ev} {rel} [GEN]" for ev, rel in chunk]
            enc = tokenizer(
                queries,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(device)
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    decoder_start_token_id=None,
                    num_beams=5,
                    num_return_sequences=5,
                    max_new_tokens=24,
                )
            decoded = tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            assert len(decoded) == len(chunk) * 5
            for i, (ev, rel) in enumerate(chunk):
                outs = decoded[i * 5 : (i + 1) * 5]
                cache[(ev, rel)] = outs
                out_f.write(json.dumps({"event": ev, "rel": rel, "outputs": outs}, ensure_ascii=False) + "\n")
            if (start // batch_size) % 20 == 0:
                out_f.flush()

    elapsed = time.time() - t0
    print(f"[COMET] 批量推理完成: {len(pending)} 条查询耗时 {elapsed:.1f}s ({len(pending)/max(elapsed,1e-6):.1f} q/s)")
    return cache


def assemble_comet_dict(records, comet_cache):
    """将 COMET 缓存组装为与 loader.py:158-211 完全一致的 data_dict。"""
    data_dict = {
        "context_role": [],
        "situation": [],
        "context": [],
        "context_strategy_seqs": [],
        "strategy": [],
        "target": [],
        "emotion": [],
        "emotion_context": [],
        "utt_cs": [],
        "utt_cs_split": [],
        "situation_cs": [],
    }
    empty_rel_stats = Counter()
    zero_all_cs_samples = 0

    def lookup_cs(toks):
        ev_str = " ".join(toks)
        cs_list = defaultdict(list)
        cs_list["uttr"] = toks
        for rel in RELATIONS:
            raw_outs = comet_cache[(ev_str, rel)]
            cleaned = []
            for s in raw_outs:
                st = s.strip()
                if st == "none" or st == "." or st == "":
                    continue
                t = process_sent(_norm(s.replace(".", "")))
                if len(t) > 0:
                    cleaned.append(t)
            if len(cleaned) == 0:
                empty_rel_stats[rel] += 1
            cs_list[rel] = cleaned
        return cs_list

    for rec in records:
        utt_cs = lookup_cs(rec["last_uttr"])
        utt_cs_split = [lookup_cs(ev) for ev in rec["split_events"]]
        situation_cs = lookup_cs(rec["situation"])

        # 检查是否出现全部 4 个认知关系在整句与子句中同时为空的极端样本
        total_nonreact = sum(len(utt_cs[r]) for r in RELATIONS[:4]) + sum(
            len(sc[r]) for sc in utt_cs_split for r in RELATIONS[:4]
        )
        if total_nonreact == 0:
            zero_all_cs_samples += 1

        data_dict["context"].append(rec["context"])
        data_dict["context_role"].append(rec["context_role"])
        data_dict["context_strategy_seqs"].append(rec["context_strategy_seqs"])
        data_dict["situation"].append(rec["situation"])
        data_dict["emotion"].append(rec["emotion"])
        data_dict["emotion_context"].append(rec["emotion_context"])
        data_dict["strategy"].append(rec["strategy"])
        data_dict["target"].append(rec["target"])
        data_dict["utt_cs"].append(utt_cs)
        data_dict["utt_cs_split"].append(utt_cs_split)
        data_dict["situation_cs"].append(situation_cs)

    return data_dict, dict(empty_rel_stats), zero_all_cs_samples


def build_conceptnet_for_esconv(data_dir, dataset, vocab, assertions_gz_path, stop_words):
    """
    流式解析 ConceptNet 5.7 压缩断言文件，构建 ESConv 专属 ConceptNet.json 与 ConceptNet_VAD_dict.json。
    完全复现 CASE 论文 VAD 强度与情感间隙打分公式，并修正原版未正确按得分降序写入的细节。
    """
    from src.models.common import gen_embeddings
    from src.utils.data.loader import emotion_intensity

    out_ranked = os.path.join(data_dir, dataset, "ConceptNet_VAD_dict.json")
    out_raw = os.path.join(data_dir, dataset, "ConceptNet.json")
    if os.path.isfile(out_ranked) and os.path.isfile(out_raw):
        print(f"[ConceptNet] 检测到已生成的 {out_ranked}，跳过重建。")
        return

    assert os.path.isfile(assertions_gz_path), f"未找到 ConceptNet 断言文件: {assertions_gz_path}"
    vad = json.load(open(os.path.join(data_dir, "VAD.json"), "r", encoding="utf-8"))
    embeddings = gen_embeddings(vocab)
    word2index = vocab.word2index
    wnl = WordNetLemmatizer()

    # 仅保留存在于词表中的纯英文单词的 GloVe 向量，预先 L2 归一化以大幅加速余弦相似度计算
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    emb_norm = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)

    concept_dict = {}
    relation_dict = Counter()
    kept_edges = 0

    opener = gzip.open if assertions_gz_path.endswith(".gz") else open
    print(f"[ConceptNet] 开始流式解析: {assertions_gz_path} (ESConv vocab={vocab.n_words})")
    with opener(assertions_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > 0 and i % 5_000_000 == 0:
                print(f"[ConceptNet] 已扫描 {i:,} 行，命中词表边数={kept_edges:,}，概念词={len(concept_dict):,}")
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            start_uri, end_uri = parts[2], parts[3]
            if not (start_uri.startswith("/c/en/") and end_uri.startswith("/c/en/")):
                continue
            s_seg = start_uri.split("/")
            e_seg = end_uri.split("/")
            if len(s_seg) < 4 or len(e_seg) < 4:
                continue
            c1 = wnl.lemmatize(s_seg[3])
            c2 = wnl.lemmatize(e_seg[3])
            if c1 == c2 or not (c1.isalpha() and c2.isalpha()):
                continue
            if c1 not in word2index or c2 not in word2index:
                continue

            try:
                weight = float(json.loads(parts[4]).get("weight", 0.0))
            except Exception:
                continue
            if weight < 1.0:
                continue

            rel_seg = parts[1].split("/")
            relation = rel_seg[2] if len(rel_seg) >= 3 else parts[1]
            if relation not in word2index:
                relation_dict[relation] += 1

            idx1, idx2 = word2index[c1], word2index[c2]
            c1_c2_sim = float(torch.dot(emb_norm[idx1], emb_norm[idx2]).item())

            v1, a1, _ = vad[c1] if c1 in vad else (0.5, 0.0, 0.5)
            v2, a2, _ = vad[c2] if c2 in vad else (0.5, 0.0, 0.5)
            emotion_gap = 1.0 - (abs(v1 - v2) + abs(a1 - a2)) / 2.0

            if c2 not in stop_words:
                c2_vad = float(emotion_intensity(vad, c2)) if c2 in vad else 0.0
                score = c2_vad + emotion_gap
                concept_dict.setdefault(c1, {})[c2] = [relation, c2_vad, c1_c2_sim, weight, emotion_gap, score]
                kept_edges += 1
            if c1 not in stop_words:
                c1_vad = float(emotion_intensity(vad, c1)) if c1 in vad else 0.0
                score = c1_vad + emotion_gap
                concept_dict.setdefault(c2, {})[c1] = [relation, c1_vad, c1_c2_sim, weight, emotion_gap, score]
                kept_edges += 1

    print(f"[ConceptNet] 解析完成：命中概念词数={len(concept_dict):,}，有效有向边={kept_edges:,}")
    with open(out_raw, "w", encoding="utf-8") as f:
        json.dump(concept_dict, f)
    with open(os.path.join(data_dir, dataset, "relation.json"), "w", encoding="utf-8") as f:
        json.dump(relation_dict.most_common(), f)

    rank_concept = {}
    for w, nbrs in concept_dict.items():
        ordered = sorted(nbrs.items(), key=lambda x: x[1][5], reverse=True)
        rank_concept[w] = [[nb, v[0], v[1], v[2], v[3], v[4], v[5]] for nb, v in ordered]
    with open(out_ranked, "w", encoding="utf-8") as f:
        json.dump(rank_concept, f, indent=2)
    print(f"[ConceptNet] 已写入按 VAD+情感间隙降序排序的概念字典: {out_ranked}")


def ensure_nltk_resources():
    """启动前集中检查并补齐预处理必需的 NLTK 语料与词性标注器，避免中途失败。"""
    required = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path_key, pkg in required:
        try:
            nltk.data.find(path_key)
        except LookupError:
            print(f"[NLTK] 缺少 {pkg}，正在自动下载...")
            nltk.download(pkg, quiet=True)
    # 兼容 NLTK >= 3.8.2 的 punkt_tab 与 averaged_perceptron_tagger_eng
    for opt_key, opt_pkg in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(opt_key)
        except LookupError:
            try:
                nltk.download(opt_pkg, quiet=True)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="构建干净 Session 级 ESConv dataset_preproc.p")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="ESConv")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_num_dialog", type=int, default=9)
    parser.add_argument("--concept_num", type=int, default=3)
    parser.add_argument("--split_only", action="store_true", help="仅执行 Session 级切分与泄漏审计，不跑 COMET/ConceptNet")
    parser.add_argument("--overwrite_splits", action="store_true", help="强制覆盖已有 train/dev/test_data.json")
    parser.add_argument("--comet_model", type=str, default="mismayil/comet-bart-ai2",
                        help="COMET-ATOMIC-2020 BART 本地路径或 HuggingFace 模型 ID")
    parser.add_argument("--comet_batch_size", type=int, default=128)
    parser.add_argument("--comet_fp32", action="store_true", help="强制使用 FP32 运行 COMET（默认 GPU 下使用 FP16）")
    parser.add_argument("--assertions_gz", type=str, default="data/conceptnet-assertions-5.7.0.csv.gz",
                        help="ConceptNet 5.7 压缩断言文件路径")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    esconv_dir = os.path.join(args.data_dir, args.dataset)
    os.makedirs(esconv_dir, exist_ok=True)

    train_turns, dev_turns, test_turns = build_or_load_splits(
        esconv_dir, seed=args.seed, max_num_dialog=args.max_num_dialog, overwrite_splits=args.overwrite_splits
    )
    if args.split_only:
        print("[Done] --split_only 模式完成。")
        return

    ensure_nltk_resources()
    # 同步设置全局 config 字段，复用原版 Lang 与 read_concept_vad
    from src.utils.config import config
    from src.utils.data.loader import Lang, read_concept_vad

    config.data_dir = args.data_dir
    config.dataset = args.dataset
    config.concept_num = args.concept_num

    comet_cache_pkl = os.path.join(esconv_dir, "dataset_comet_preproc.p")
    final_pkl = os.path.join(esconv_dir, "dataset_preproc.p")
    stop_words = set(stopwords.words("english"))
    emotion_lexicon = json.load(open(os.path.join(args.data_dir, "NRCDict.json"), "r", encoding="utf-8"))[0]

    if os.path.isfile(comet_cache_pkl):
        print(f"[COMET] 检测到已有 {comet_cache_pkl}，直接加载。")
        with open(comet_cache_pkl, "rb") as f:
            data_tra, data_val, data_tst, vocab = pickle.load(f)
        comet_audit = {"loaded_from_existing_pickle": True}
    else:
        vocab = Lang({
            config.UNK_idx: "UNK",
            config.PAD_idx: "PAD",
            config.EOS_idx: "EOS",
            config.SOS_idx: "SOS",
            config.USR_idx: "USR",
            config.SYS_idx: "SYS",
            config.CLS_idx: "CLS",
            config.KG_idx: "KG",
            config.SEP_idx: "SEP",
        })
        tr_rec, ev_tr = pretokenize_split(train_turns, vocab, stop_words, emotion_lexicon)
        dv_rec, ev_dv = pretokenize_split(dev_turns, vocab, stop_words, emotion_lexicon)
        te_rec, ev_te = pretokenize_split(test_turns, vocab, stop_words, emotion_lexicon)
        all_events = ev_tr | ev_dv | ev_te
        print(f"[Vocab] ESConv 词表大小 = {vocab.n_words}, 唯一 COMET 事件数 = {len(all_events)}")

        cache_jsonl = os.path.join(esconv_dir, "comet_cache.jsonl")
        comet_cache = run_batched_comet(
            all_events,
            comet_model_path=args.comet_model,
            cache_jsonl_path=cache_jsonl,
            device=args.device,
            batch_size=args.comet_batch_size,
            fp16=not args.comet_fp32,
        )
        data_tra, emp_tr, z_tr = assemble_comet_dict(tr_rec, comet_cache)
        data_val, emp_dv, z_dv = assemble_comet_dict(dv_rec, comet_cache)
        data_tst, emp_te, z_te = assemble_comet_dict(te_rec, comet_cache)
        comet_audit = {
            "comet_model": args.comet_model,
            "unique_events": len(all_events),
            "empty_rel_counts": {"train": emp_tr, "dev": emp_dv, "test": emp_te},
            "zero_all_nonreact_samples": {"train": z_tr, "dev": z_dv, "test": z_te},
        }
        with open(comet_cache_pkl, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
        print(f"[COMET] 已保存中间文件: {comet_cache_pkl}")

    build_conceptnet_for_esconv(args.data_dir, args.dataset, vocab, args.assertions_gz, stop_words)
    data_tra, data_val, data_tst, vocab = read_concept_vad(
        args.data_dir, args.dataset, data_train=data_tra, data_val=data_val, data_test=data_tst, vocab=vocab
    )
    with open(final_pkl, "wb") as f:
        pickle.dump([data_tra, data_val, data_tst, vocab], f)

    dataset_audit = {
        "dataset": args.dataset,
        "vocab_size": vocab.n_words,
        "sizes": {"train": len(data_tra["target"]), "dev": len(data_val["target"]), "test": len(data_tst["target"])},
        "keys": list(data_tra.keys()),
        "comet_audit": comet_audit,
        "final_pkl_sha256": sha256_file(final_pkl),
    }
    audit_out = os.path.join(esconv_dir, "dataset_audit.json")
    with open(audit_out, "w", encoding="utf-8") as f:
        json.dump(dataset_audit, f, indent=2, ensure_ascii=False)
    print(f"[Done] 最终 ESConv 数据集已落盘: {final_pkl} (vocab={vocab.n_words}, sha256={dataset_audit['final_pkl_sha256'][:16]}...)")


if __name__ == "__main__":
    main()
