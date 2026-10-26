# V10 Trial 1 v1.2 评测协议复核

日期：2026-10-26。范围：用户在云端开跑前补入的历史参照、全量生成与判读规则。仅修改文档，未运行新的训练、配对补评或全量生成。遵循实验审计规则 v2；不改规则文件。

## 结论与已核实事实

- T11-B 仍为唯一直接对照，T10 与 T11（seed13）为必须报告的历史参照；训练组、底座、双门槛 `32.56 / 42.21%`、选点规则与本地验收容差不变。
- 三行历史指标与原始报告相符：[T10](../../../results/cloud_archive/v9_trial10/v9_trial10_eval.txt#L5)、[T11](../../../results/cloud_archive/v9_trial11/v9_trial11_seed13_eval.txt#L5)、[T11-B](../../../results/cloud_archive/v9_trial11_B_gnorm6/v9_trial11_B_gnorm6_eval.txt#L5)。T11 的裁剪启用行和 ramp=2500 见其日志第 154、92 行。工作区 `save/`、`results/` 未发现 T11 对应权重；不据此宣称其他设备也不存在。
- [`eval_pipeline.py:829`](../../../src/scripts/eval_pipeline.py#L829) 将 `--gen_samples<=0` 转为全量 Test 批数；当前 Test batch=1。实际生成数来自函数内 `count`，保存于 `evaluation_protocol.generation.evaluated_n`（第 921 行）。`--skip_test_eval --skip_manifold` 适用于仅生成补评，不产生 Test 配对数据。
- JSON 写入权重目录（第 902–903 行），文本写入 `--results_path`（第 880 行）。因此复制权重后还应显式隔离文本/日志路径；按实验与检查点分目录，避免 composite/ppl 两次补评相互覆盖。
- `main.sh` SHA256 实测为 `ceff4064c4ca196343cd795b3cc7719ccd7f116baa4c0a3d429320c6095c6964`，与 [03 验收证据](evidence/trial1_correction/verification.json) 一致。`git diff -- main.py main.sh src tests` 为空。

## 文档补正

1. “点估计更高”改为“PPL/Acc 点估计更优”，避免把更低的 PPL 写成更高；保留 T11 行和裁剪/ramp 的变量分解。
2. 配对 bootstrap / McNemar 限定于给定已训练检查点的测试样本差异，不能估计训练种子方差。只有一项区间不满足改善条件时，不再笼统写“两项区间包含 0”；逐项报告。即使满足双项条件，也限定为对指定检查点的 PPL/Acc 结论，不自动升级为所有历史模型和指标的“历史最佳”。
3. Dist-2 对样本量、长度与内容敏感，不保证单调下降。例如前两句都只有相同 bigram 时为 1/2，新增一个不同 bigram 后变为 2/3。因此保留 500/5255 分列要求，删除“必随样本量下降”的暗示。
4. 固定生成副本、报告、日志的独立路径，校验原件/副本 SHA256 并记录源码、环境、命令和实际计数；Test 配对产物另设目录，避免仅生成评测的空 Test 字段覆盖已有补评。
5. 两个限制重新区分：源码已确认 [`model.py:2518`](../../../src/models/CASE/model.py#L2518) 的 `max_grad_norm<=0` 跳过裁剪，但 V10 命名配置锁定 6.0；F1 的历史部署仍未证实。`79e0bfc` 提交时间是 2026-10-13 16:55:54 +0800，T10 日志第 102 行有 `10-14 10:25 Vocab`，但日期先后不能替代部署源码指纹，T11-B 日期较晚同样不能单独证明部署。

## 复核命令与待执行项

```bash
git -c safe.directory=E:/github/CASE diff -- main.py main.sh src tests
sha256sum main.sh
git -c safe.directory=E:/github/CASE show -s --format=fuller 79e0bfc
rg -n 'clip_grad|max_grad_norm' src main.py main.sh
rg --files --hidden --no-ignore save results | rg 'CASE_.*37\.6652'
```

本次为静态代码、原始归档和文档核对，未重复 GPU 工程验收。T10−T11-B 配对补评、T10/T11-B/V10 的全量生成以及 V10 正式训练均保持待执行状态；同机旧补评复用前仍须核对源码、环境、权重与数据指纹。GPU 空闲时的推理会占用本地算力，但不新增训练组。后续正式启动命令仍为 `bash main.sh 24G v10_trial1_sep_dec_head`。
