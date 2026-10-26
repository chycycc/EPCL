# V10 历史参照：T10 与 T11-B 同环境补评、配对统计及全量生成

日期：2026-10-26。执行基线 `v9-clean @ e97a0c0`，依据冻结方案 v1.2 和实验审计规则 v2。范围为交接 A1–A4 的本地推理及统计；没有训练、没有使用云端 GPU。T11-B 保留唯一直接对照，T10/T11 保留历史参照；双门槛 `32.56 / 42.21%` 和选点规则不变。

## 1. 结论与质量指标

**在这两个已训练检查点的同环境 Test 比较中，T10 的 PPL/Acc 点估计更优，T11-B 的 EMO_loss 点估计更优；三项差值的配对 95% 区间均包含 0。尚无三项差异的显著证据，不据此宣称等价，也不推广为训练方法或跨种子的优劣。**

口径：Test 全量 N=5255，batch=1，SequentialSampler（shuffle=False）；PPL=`exp(mean(逐样本 token CE))`，Acc=样本 micro，EMO_loss=样本平均主分类 CE。均使用已冻结的 step7499 `composite_best`，不按 Test 重新选点。

| 检查点 | 云端正式 PPL / Acc / EMO_loss | 本地 PPL（未舍入值展示9位） | 本地 Acc / 正确数 | 本地 EMO_loss |
|---|---|---:|---:|---:|
| T10 | 32.48 / 42.40% / 2.3190 | 32.474551969 | 42.397716461% (2228/5255) | 2.318990569 |
| T11-B | 32.56 / 42.21% / 2.2840 | 32.564646486 | 42.207421503% (2218/5255) | 2.283988079 |

T10 本地 PPL 两位报告为 **32.47**，与历史云端 **32.48** 不同；本地没有复现云端该舍入值，来源尚未定位，不能仅凭差异小就断言原因。历史正式值保持原样，以下配对统计仅使用此次同环境未舍入数据。两组分类预测均与各自云端归档逐条一致（5255/5255）。

T11-B 原补评缓存确实保留了 token CE、预测和样本身份，但没有逐样本 EMO CE，故本次重跑补齐。重跑 token CE 与旧缓存逐位一致（最大绝对差 **0.0**），预测差异 **0**，PPL 精确复现 `32.56464648604055`。

证据：[T10 Test](evidence/history_reference_20261026/T10_test_test_metrics.json)、[T11-B Test](evidence/history_reference_20261026/T11B_test_test_metrics.json)、[完整配对结果](evidence/history_reference_20261026/paired_comparison.json)。历史 T11 的云端质量三项仍为 `32.62 / 41.83% / 2.3396`；其无本地权重，不填入本地补评值。

## 2. T10 − T11-B 配对统计

5255 条样本身份、标签、顺序逐项一致。复用 Trial12 08 审计的方法：NumPy RandomState `seed=20261023`，10000 次逐样本配对 bootstrap，三个指标共享重抽样索引；报告边际百分位 95% 区间与完整宽度。

| 指标（T10 − T11-B） | 点差 | 配对95%区间 | 完整宽度 |
|---|---:|---:|---:|
| 样本 macro PPL | -0.090094517 | [-0.242593141, +0.057340872] | 0.299934012 |
| 样本 micro Acc（pp） | +0.190294957 | [-0.856327307, +1.236917222] | 2.093244529 |
| 样本平均 EMO_loss | +0.035002490 | [-0.004141223, +0.074269686] | 0.078410909 |

分类配对四格：双方正确 **1828**，仅 T10 正确 **400**，仅 T11-B 正确 **390**，双方错误 **2637**。精确双侧 McNemar **p=0.748835480580**（未作多重校正）。

验证：独立整数二项系数求和与 SciPy McNemar p 差 <1e-12；bootstrap 分批大小 100 与 50 得到全部10000次、全部三指标的分布逐位一致。上述区间条件于固定模型和样本独立假设，不覆盖对话内相关性、训练种子方差或历史训练环境差异；本轮不增加训练种子。T10/T11-B 之间还存在裁剪和 Arc ramp 联合变化，不能把点差归因给其中某一项。

## 3. 生成四指标：500 与 5255 分列

两列都按 **Greedy Dist-2 / Greedy Unique / Beam Dist-2 / Beam Unique** 排列，单位均为%。500 列来自历史云端归档；5255 列来自本次本地同环境全量推理。均为 Test 顺序、batch=1、max_dec_step=40、Beam k=5；两种样本量不可相减或混作同口径胜负比较。

| 检查点 | 历史云端前 N=500 | 本地全量 N=5255 |
|---|---|---|
| T10 | 18.14 / 62.20 / 13.63 / 41.80 | 4.35 / 36.59 / 2.90 / 18.10 |
| T11（seed13） | 18.94 / 62.00 / 13.16 / 39.00 | 未测：本地无对应权重 |
| T11-B | 18.19 / 68.00 / 14.61 / 42.60 | 4.66 / 38.61 / 2.98 / 18.06 |

同口径全量差值（T10 − T11-B），按上述四指标顺序为 **-0.31 / -2.02 / -0.08 / +0.04 pp**。这些是已生成语料的描述性指标，不当作训练效果的独立显著性证据；不单挑其中一项宣称整体生成质量改善。

两路核采样（T=0.7/0.8，top-p=0.9）保留在原始 JSON，仅作辅助记录。[T10 生成结果](evidence/history_reference_20261026/T10_generation_eval_metrics.json)、[T11-B 生成结果](evidence/history_reference_20261026/T11B_generation_eval_metrics.json)。实际 `generation.evaluated_n` 均为5255，Test/流形在生成调用中均显式 skipped；没有用空 Test 字段覆盖配对补评。

## 4. 运行与归档完整性

- 本地环境：Python3.8.20、PyTorch1.12.1+cu113、CUDA11.3、cuDNN8302、RTX3050Ti 4GB、NumPy1.24.4、SciPy1.9.3、sklearn1.0.2、NLTK3.5；两个模型采用相同源码、数据与评测参数。环境、源码/数据/原始归档SHA见 [input_manifest.json](evidence/history_reference_20261026/input_manifest.json)，分词器资源另有[指纹](evidence/history_reference_20261026/tokenizer_environment.json)。
- 原权重复制到独立的“实验/检查点”目录，副本与原件 SHA256 相同。Test 与 generation 分目录；每次运行显式指定准确的权重文件、报告和日志路径，未对 cloud_archive 原件执行评测。全部原始归档文件、生产源码、数据缓存运行前后 SHA256 一致。
- Test 模型的全部 state tensor 均逐位恢复，并检查全部逐样本 CE 有限。11项原验收源码/测试文件不因本次推理发生改变；本轮未改 `main.sh`、训练代码、PCGrad 或累加路径。
- 实测耗时（含加载）：test_T10：3.7 分钟；test_T11B：3.6 分钟；generation_T10：66.9 分钟；generation_T11B：65.5 分钟。两组 Test 后串行执行两组生成，没有并发争抢4GB显存。
- T10 检查点 SHA256：`b7c0d0fb7c1e5a41a2133e5dde9405370379cf35e4a0d48101fad6f9f358f0ce`；T11-B：`aa559e7e912faa1b12a85e0c70ad01a5798cf80fa48df789ca30783b085fbe7b`。完整验证与小型证据哈希见 [verification.json](evidence/history_reference_20261026/verification.json)。

完整派生产物：`results/v10/history_reference_20261026/`（Test/统计/源码快照），`results/v10/gen_full5255/<实验名>/<检查点名>/`（生成及权重副本）。小型逐样本 NPZ、bootstrap 分布、指标、命令与清单归档于 `docs/v10/reports/evidence/history_reference_20261026/`；实验权重、临时脚本及大日志不提交 Git。

## 5. 已执行命令与下一步

临时脚本仅在 `scratch/v10_trial1/`，源码快照保留于派生目录；脚本拒绝覆盖既有结果。实际完整生成命令及加载参数见 [T10 command](evidence/history_reference_20261026/T10_generation_command.json)、[T11-B command](evidence/history_reference_20261026/T11B_generation_command.json)，核心参数为 `eval_pipeline.py --gen_samples 0 --skip_test_eval --skip_manifold --batch_size 1 --use_arc_margin`。

```text
E:/Dev/condaData/envs_dirs/cem_env/python.exe scratch/v10_trial1/history_reference_eval.py prepare
E:/Dev/condaData/envs_dirs/cem_env/python.exe scratch/v10_trial1/history_reference_eval.py queue
E:/Dev/condaData/envs_dirs/cem_env/python.exe scratch/v10_trial1/analyze_history_reference.py
E:/Dev/condaData/envs_dirs/cem_env/python.exe scratch/v10_trial1/finalize_history_reference.py
```

A1–A4 已完成；本报告和总账完成 A5。用户已完成 B 并回传 V10 云端归档；C 的完整审计另见报告 06。本报告只记录历史参照补评，不预定 Trial2、不切换底座。
