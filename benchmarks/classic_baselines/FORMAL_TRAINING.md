# 四模型正式训练

2026-10-30 日志更新：控制台采用进度条与具名指标，增加 TensorBoard；当前运行日志可只读实时转换，见 [LOGGING_UPDATE.md](LOGGING_UPDATE.md)。

配方确定日期：2026-10-28。四模型依次为 CASE、Multi-TRS、CEM、KEMP；采用各自原生结构与损失，是统一数据下的系统比较。正式路径遵循规则 v3：Git 记录代码版本，初始化输出一行 `[Architecture Fingerprint]`，直接运行训练与评测。

## 已确定的配方

- batch=16、accum=1、FP32、seed=13；GloVe 使用云端现有文件。
- CASE 保留 4 轮 BOW 预训练，预训练后保留 Adam 状态和 Noam 步数。40,250 条 Train、保留尾批，共 10,064 次预训练更新。其他三模型直接进入主训练。
- Adam/Noam 沿用原生实现：CASE warmup=12000，其余 8000。保留模型原生 loss，未加入 CASE-EPCL 的 PCGrad、裁剪、ramp 或其他组件。
- 主训练每 2000 次更新做完整 Valid，batch=16、shuffle=False。选择指标沿用原版 `exp(mean(batch token CE))`。
- 第 14000 次主训练更新首次参与选优。PPL 小于或等于历史最优时保存并清零 patience，连续 3 个验证点严格变差则早停。
- 主训练最多 30000 次更新，CASE 预训练另计。到上限停止标为 `budget_limit`，不是收敛声明。
- Train 按 `seed + epoch` 生成独立随机排列，保留尾批。检查点记录 epoch、批次游标、模型/Adam/Noam/随机数和动态词频，续训从最近保存位置继续。
- 完整 Test：N=5255、batch=1、shuffle=False，输出样本 macro PPL、token micro PPL、样本 micro Acc 与平均 EMO CE。Greedy/Beam 只生成 Test 顺序前 500 条，beam=5、max_dec_step=40。

## 上传和启动

把 `baseline_training_v1.tar.gz` 上传到现有云端目录，解压后启动第一组 CASE。使用已经通过烟测的同一 Python 环境。

```bash
cd /root/autodl-tmp/baseline_reproduction/classic_baselines_bundle
tar --keep-old-files -xzf baseline_training_v1.tar.gz
bash baseline_training_v1/train.sh case_original
```

这条 `train.sh` 命令会启动正式训练，随后自动评测最优检查点。新的工具目录复用既有 `workspace/` 和 `data/`；GloVe 默认路径为 `/root/autodl-tmp/CASE/vectors/glove.6B.300d.txt`。

CASE 完成后依次执行（不并行抢占 GPU）：

```bash
bash baseline_training_v1/train.sh multi_trs
bash baseline_training_v1/train.sh cem_original
bash baseline_training_v1/train.sh kemp
```

断线前建议在云端已有 tmux 会话中启动。中断后续训，或单独重跑最佳模型评测：

```bash
bash baseline_training_v1/train.sh case_original --resume
bash baseline_training_v1/train.sh case_original --evaluate-only
```

输出目录已存在时，新训练直接报错；续训需显式 `--resume`。如需另一个运行目录，指定 `BASELINE_RUN_DIR=/实际新目录`。其他可选环境变量为 `BASELINE_BUNDLE_ROOT`、`BASELINE_PYTHON`、`BASELINE_GLOVE`；NLTK 可通过已有 `NLTK_DATA` 指定。

## 结果在哪里

默认每模型写入 `runs/formal_<model>/`，控制台完整日志为相邻的 `runs/formal_<model>.log`。

- `last.pt`：训练恢复状态；预训练每 2000 次更新及预训练结束保存，主训练每次验证后保存。
- `ppl_best.pt`：用于最后评测的最佳 Valid PPL 权重。基线本身没有 composite 选择目标，不虚构 composite 检查点；CASE-EPCL 对照表仍按 v3 并列其双 best。
- `run.json`、`train.jsonl`、`valid.jsonl`、`training_result.json`：有效配置、训练轨迹、全部验证点和停止原因。
- **`eval_metrics.json`**：本次正式报告的指标来源。
- `test_records.jsonl`、`generation_500.jsonl`：逐样本指标与生成文本。

第一组完成后回传 `eval_metrics.json`、`training_result.json`、`valid.jsonl` 和完整训练日志即可。权重留在云端；后续按实际需要下载。

本轮实现验证仅运行 4 项训练逻辑测试与 Shell 语法检查，覆盖选优/早停、随机批次恢复、Valid 聚合、含预训练的 Adam/Noam 续训；未启动四模型正式训练。
