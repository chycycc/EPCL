# 四模型独立训练烟测入口与本地验收

日期：2026-10-26。分支核验为 `v9-clean`。依据实验审计规则 v2；不修改规则文件。四模型云端 GPU 推理通过、Python 3.10.8 / PyTorch 2.1.2+cu121 / RTX 4090D 来自用户本轮回传；本报告新增实测为本地 Python 3.8.20 / torch 1.12.1+cu113、**CPU**，不标为 GPU 训练验收。

## 1. 改动与保护边界

先读取了旁支未提交的 `native_runtime.py`、`cloud_preflight.py`、`cloud_target.json`、两份测试和云端说明差异。保留 CASE `accuracy_score` 转 numpy.float64 的 `.item()` 兼容、GPU `4090 D` 空格兼容和 torch 2.1.2 目标。当前修改只在 `benchmarks/classic_baselines/` 与基线归档文档中；已有其他改动保留。

- `native_runtime.py:68` 新增可选 `batch_size` 构造参数，缺省仍为 16，仅训练 smoke 显式传入 2。原推理接口与旁支补丁保留。
- `glove_audit.py:5` 固定云端原始 GloVe SHA；`:14` 对原生加载及 Xavier 初始化后的全部命中行逐位核验。
- `native_train_smoke.py:41` 调用各自原生 `train_one_batch(..., train=True)`，观测实际 backward/Adam step/zero_grad、梯度、权重更新；不重写原生目标函数或优化器。
- `training_state.py:47` 保存完整状态，`:78` 在新模型中严格恢复并逐位核验，`:110` 对重放下一步作结构及数值比较。
- `run_training_smoke_queue.py:16` 串行执行 8 个独立 worker（每模型 initial/resume 两进程），硬编码烟测步数，没有 epoch/正式训练预算参数。
- `pack_training_smoke_tools.py` 生成新的版本化工具目录，复用云端原 workspace/data；不覆盖既有 tools，也不打包 GloVe、缓存、实验权重。

按历史保护清单重算的 **54 个文件 SHA256 全部未变**，包含 CASE-EPCL 生产代码、根 tests、规则与原 CEM 相关源码。另对本轮开始时快照复核，旁支的 cloud_preflight/cloud_target 与测试文件原字节保留。证据见 [verification.json](evidence/training_smoke_20261026/verification.json)。

## 2. 检查点与验收定义

原生 Noam `state_dict()` 仅返回 Adam 状态（例如原版 CASE `src/models/common.py:822`），遗漏 `_step/_rate/warmup/factor/model_size`。专用烟测检查点额外保存这些字段、模型权重、Adam、Python/NumPy/torch CPU/CUDA RNG、动态 word_freq 与 criterion weight/true_dist、训练模式及实际计数。

每模型先做两次主训练更新后保存；继续第三次取得参考，然后新进程构造、完整恢复并重放第三次。CASE 额外先执行一次 BOW 预训练更新，只验证此分支，不冒充完整四轮预训练或正式预训练配方。恢复须逐位相等；续步浮点 tensor 容差预先固定为 `atol=1e-6, rtol=1e-5`，离散/RNG/Noam/词频等状态精确比较。此容差不涉及 V10 数学验收。

FP32、seed=13、batch=2、accum=1，固定 Train 首 6 条，不用 Test 调参。每个主训练步骤检查 embedding、分类头、decoder 梯度非零；所有现存梯度、权重和 Adam 状态有限；embedding、分类头实际更新。记录真实反向标量，不把日志中的 PPL 换作优化目标。

恢复后再只读检查一条 Valid 并断言全状态不变。**Test N=0、生成 N=0**。没有正式训练、检查点选优或指标胜负结论。

## 3. 实测结果

独立工具合同测试 **29/29 通过**，含新增 6 项训练状态/GloVe 测试；上传包解压后的同一套 29 项再次通过。覆盖完整 Adam/Noam/动态 buffer/RNG 恢复和 dropout 续步重放、来源不符先拒绝、Noam/RNG/Adam 篡改可检出、非有限/超容差/结构错误拒绝，以及 GloVe 哈希及所有命中行核验。

原始运行：`results/benchmarks/classic_baselines/training_smoke_20261026/local_r2/`。小证据完整归档于 [evidence/training_smoke_20261026/](evidence/training_smoke_20261026/queue.json)，含 8 份日志、环境及 metadata。各模型结果如下：

| 模型 | 实际更新总数（含重放） | 每次主训练发生变化的参数张量数 | 实测最大单步参数变化 | 完整恢复 | 下一步状态/目标误差 |
|---|---:|---:|---:|---|---|
| CASE | 5 | 144 | 2.38418579e-7 | 逐位相等 | 0 / 0 |
| Multi-TRS | 4 | 40 | 2.45869160e-7 | 逐位相等 | 0 / 0 |
| CEM | 4 | 102 | 2.45869160e-7 | 逐位相等 | 0 / 0 |
| KEMP | 4 | 55 | 2.45869160e-7 | 逐位相等 | 0 / 0 |

合计 **17 次真实优化器调用**；连续轨迹为 CASE 4 次、其余 3 次，剩余各一次为新进程恢复重放，不重复计为训练进度。所有现存梯度、优化目标、模型和优化器状态有限；四组只读 Valid 检查状态不变。

四模型均读取原始 GloVe 400,000 行；24,647 词表中命中 **19,481 行（79.04%）**，未命中 5,166 行保留原生随机初始化。全部命中行与 GloVe float32 值逐位一致，最大误差 **0**；四组初始 embedding SHA 相同：`10d1af93b9cecae2afea8e417109ca5b6b5ace785190fc3f464b1ee014ec66ef`。

本地原 GloVe 只读；在 scratch 新建副本去掉末尾 `<unk>` 一行（2,982 bytes）后，全文件 SHA 恰为用户提供的云端值 `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`。原文件未修改。

本地 Adam `zero_grad` 签名实测为 `set_to_none=False`；首步所有 grad 为 None，后续原进程可见保留的零 tensor。新进程从 None 梯度续步仍达到上述逐位结果。云端默认行为由新 worker 继续实际记录，不以本地默认值推断。Noam warmup 为 CASE 12000，其余 8000；保留原生 tiny warmup 初始学习率，没有为让更新显著而放大学习率。

首次 `local_r1` 在读取 GloVe 时遇 Windows 默认 GBK 解码错误，未发生优化器更新；已由父队列为 worker 设置 `PYTHONUTF8=1`，不改原生模型文件。失败日志保留于 results，不计为通过。`local_r2` 全队列通过。

## 4. 增量包及云端下一步

本地上传文件：

```text
results/benchmarks/classic_baselines/training_smoke_20261026/cloud/training_smoke_tools_20261026_v1.tar.gz
```

大小 **48,354 bytes**，SHA256：`ceb1c2cacf9758d99ee5b1bec345632761309fd2db698a07af537225599b4ebe`。逐文件清单见 [incremental_archive_manifest.json](evidence/training_smoke_20261026/incremental_archive_manifest.json)；解压哈希和包内测试均已验证。

上传到 `/root/autodl-tmp/baseline_reproduction/classic_baselines_bundle/`，按 [TRAINING_SMOKE_README.md](../../../benchmarks/classic_baselines/TRAINING_SMOKE_README.md) 执行。它创建独立 `training_smoke_tools_20261026_v1/`，输出到新的 `runs/cloud_training_smoke_v1/`。继续使用已通过推理验收的环境快照及云端 GloVe，不需要重新传数据、原生源码或词向量。

通过条件为队列 `all_passed=true`、实际更新总数 17，且每模型恢复/续步/只读检查通过。回传 queue、8 份日志、metadata/environment；权重先留云端。**本地 CPU 通过不替代 PyTorch 2.1.2 GPU 训练验收**；完整预训练、正式 batch/训练预算/选点与早停配方尚待冻结。

复核命令（在 CASE 根目录）：

```bash
python -m unittest discover -s benchmarks/classic_baselines/tests -v
python benchmarks/classic_baselines/run_training_smoke_queue.py --workspace-root benchmarks/classic_baselines/workspace/preparation_20261026_v2 --data-dir results/benchmarks/classic_baselines/phase2_20261026/data --output-dir results/benchmarks/classic_baselines/training_smoke_20261026/new_cpu_run --embedding-file scratch/classic_baselines_training_smoke/glove.6B.300d.cloud_original.txt --device cpu --batch-size 2
```
