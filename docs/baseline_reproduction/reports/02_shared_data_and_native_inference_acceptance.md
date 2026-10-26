# 统一数据与四模型 CPU 推理验收

日期：2026-10-26。前序审计提交为 `200bcb1`。本轮推进统一数据、原生推理导出、搜索和云端准备包；**没有训练任何模型，没有执行云端 PyTorch 2.1 验收**。生产代码、V10 方案和 sibling CEM 工程保持原样。

## 1. 数据协议与来源

共享包由当前 CASE 的 `data/ED/dataset_preproc.p` 只读导出。样本、分词、24,647 词表、COMET 与 ConceptNet/VAD 特征均固定到这份已存在的缓存；不是重新训练或调用 COMET。CASE 原缓存与旧 CEM 缓存未改写。

这是“原生模型结构 + 统一数据协议下复现”，不是作者原始预处理缓存的逐位复刻。CEM/Multi-TRS 的输出词表容量由其旧缓存的 23,712 改为共享 24,647；KEMP 的知识输入来自 CASE 的已有概念/VAD 缓存，而非声称下载了作者 KEMP 缓存。后续主表必须标明这一复现设定，不能与论文原表数值混作同源测量。

- CEM 的常识字典按 `xIntent/xNeed/xWant/xEffect/xReact` 显式转换为五路列表，模型仍使用自己的知识编码器与损失。
- KEMP 保留原生 Dataset、概念图、情绪依赖 decoder、pointer generator、attention loss。`USR/SYS` 只重命名为 `[USR]/[SYS]`，ID 不变、不增加词表项；特殊 ID 按共享词表对齐。
- 全量核验 context/concepts/VAD 的 turn、token、概念长度；常识来源 `uttr` 等于 ED 最后一个 context turn。另核验 CEM 原生过滤后 `xReact` 没有空序列。
- 共享词表构成沿用已有 CASE 缓存，不根据新模型 Test 成绩重建或筛选；该缓存原先的词表构造规则也随复现声明保留。

| split | 实际样本数 | context token | concept token | target token（不含 EOS） | 三类 token 的 OOV |
|---|---:|---:|---:|---:|---:|
| Train | 40,250 | 1,389,160 | 1,120,410 | 544,561 | 均为 0 |
| Valid | 5,734 | 215,184 | 173,089 | 81,248 | 均为 0 |
| Test | 5,255 | 212,513 | 169,497 | 75,582 | 均为 0 |

目标编码 SHA256：`ba4bf68d84cf15a88c5055c37e96cc3027fb75de886679fd7c69e4380a8989bb`。共享数据文件 SHA256：`2e2ac956f53df48ef9b54f22d3ae1a0e9f752ee8432db22e105f523e31a88b2c`。完整来源与计数见 [data_manifest.json](evidence/phase2_20261026/data_manifest.json)、[knowledge_edge_cases.json](evidence/phase2_20261026/knowledge_edge_cases.json)。

## 2. 原生导出和搜索

每模型以独立进程/cwd 导入自身的上游源码，保存并核验实际模块路径；没有导入生产 CASE-EPCL 的 `src`。不导入/执行上游训练 `main`：KEMP 的 parser 函数通过 AST 单独读取，其模块内 seed=0 重置后再显式设置 seed=13。

教师强制评测用 hooks 捕获原生 generator 的 log 分布和原生分类头 logits，从真实目标 ID 计算未平滑、未加权 NLL/分类 CE。NLL 排除 PAD、包含 EOS；核对实际输出支持恰为 24,647，没有 pointer 扩展 OOV 项。记录原生返回 CE 与双精度求和结果的浮点误差，不混入辅助目标。

评测包装恢复 Python/NumPy/torch RNG、`word_freq`、criterion 权重和临时分布；不调用优化器更新。检查点恢复为 strict=True，并要求源 commit、目标编码指纹以及 trained/untrained 标记；随机检查点不能用全量导出伪装成训练结果。

生成上下文只从原生 `decoder_greedy` 路径捕获，不复用教师强制的 decoder 上下文。共同搜索累计 log 概率、长度惩罚 0，预算包含 EOS；文本规范固定为 NFKC + strip + TreebankWordTokenizer，保留大小写。每条 smoke 核对：

1. 共同 Greedy 等于相同生成 token 预算下的原生 Greedy；原生 `max_dec_step+1` 调用时传 `budget-1`。
2. 将目标 token 替换为 SOS 后，原生生成的首步分布逐位不变。这是实际检查，不代表自由生成情绪准确率。
3. 整个推理前后模型 state_dict 指纹一致，实际原始优化器 step 调用次数为 0。

共同 Beam 不调用 KEMP 旧的 Translator 兼容入口。CASE-EPCL 历史 Greedy/Beam 数值仍需按同一搜索和文本协议重导出/复算才能进新比较表；不修改既有 V10 正式评测协议。

## 3. 实测结果与限制

实际本地环境 Python 3.8.20 / PyTorch 1.12.1+cu113，device=CPU；各组随机初始化、不载入 GloVe、seed=13、FP32。每组 Test 前 2 条，最多生成 3 token、beam=2。**这些是接口 smoke，不是正式基线性能，也没有反向或训练更新。**

| 模型 | N | 保存/扰动后恢复 | 推理状态不变 | 实际 optimizer 更新 | 最大原生 CE 误差 |
|---|---:|---|---|---:|---:|
| 原版 CASE | 2 | 逐位通过 | 通过 | 0 | 3.4060e-7 |
| Multi-TRS | 2 | 逐位通过 | 通过 | 0 | 1.0218e-6 |
| 原版 CEM | 2 | 逐位通过 | 通过 | 0 | 2.7248e-7 |
| KEMP | 2 | 逐位通过 | 通过 | 0 | 1.7030e-7 |

这里的误差是 float32 原生均值与 float64 token 求和之间的差异，检查容差为 1e-5；不是 V10 的梯度等价性误差或对其冻结容差的修改。所有原生 Greedy 一致、目标替换检查均通过。四份 JSONL 再经独立 `score_records.py` CLI 复算，指标字段完全一致。

证据见 [queue.json](evidence/phase2_20261026/queue.json)、各模型 metadata/records 和 [independent_rescore.json](evidence/phase2_20261026/independent_rescore.json)。原生实际 Noam warmup 为 CASE=12,000、其余=8,000，Adam betas 均为 (0.9, 0.98)；完整生效配置逐模型记录，不能用命令行名义 lr 推断 Noam 的实际曲线。

工具回归 **22 项通过**，包含数据关系布局、KEMP 词表同构、OOV/维度拒绝、RNG/频率恢复、检查点来源拒绝、Beam 保留完成序列、Greedy tie/EOS 预算以及包哈希/路径检查。打包后的工具副本也重新运行同一套测试通过。前序适配调试出现的 collate 接口错误和并行读大缓存的 MemoryError 已通过接口修正、串行运行解决；失败输出保留，未作为通过证据。

## 4. 云端准备包

本地文件：`results/benchmarks/classic_baselines/phase2_20261026/cloud/classic_baselines_preparation_20261026_v3.tar.gz`。

- 大小 **65,689,053 bytes**（约 66 MB），110 个文件；包含独立源码、工具、统一数据及指纹，不含模型权重与训练日志。
- SHA256：`40cbb97d8d3c33f71aad2a981e5dc8d34fb6494d3fe355cffe9dc2be0e5ae256`。
- 逐项流式校验压缩包文件指纹，且只解出工具后完成 22 项回归。见 [archive_verification.json](evidence/phase2_20261026/archive_verification.json)。完整打包清单见 [cloud_bundle_manifest.json](evidence/phase2_20261026/cloud_bundle_manifest.json)。GloVe 只记录外部文件 SHA256，没有重复打包约 1GB 的词向量。
- 云端步骤见 [CLOUD_README.md](../../../benchmarks/classic_baselines/CLOUD_README.md)：先包校验，再实际 CASE-EPCL 环境快照、工具测试、串行推理 smoke。这里没有正式训练启动命令。

## 5. 待执行项与复核命令

仍需指定云端 Python 3.10 / torch 2.1 / CUDA 12.1 / 4090D 实测；还需 GloVe 初始化、训练反向/更新路径、有效训练预算/选点/早停配方冻结，以及 CASE-EPCL 参考生成协议对齐。当前 `cloud_ready=false`。用户暂时无需提供数据或修改代码；资源可用时上传一个准备包即可先执行云端推理检查。

```bash
python -m unittest discover -s benchmarks/classic_baselines/tests -v
python benchmarks/classic_baselines/build_data_bundle.py --case-root E:/github/CASE --output-dir results/benchmarks/classic_baselines/new_shared_data
python benchmarks/classic_baselines/run_smoke_queue.py --workspace-root benchmarks/classic_baselines/workspace/preparation_20261026_v2 --data-dir results/benchmarks/classic_baselines/phase2_20261026/data --output-dir results/benchmarks/classic_baselines/new_cpu_smoke --device cpu --nltk-data C:/Users/chy/AppData/Roaming/nltk_data
git -c safe.directory=E:/github/CASE diff -- main.py main.sh src tests
```

原缓存、54 个受保护文件和当前脚本指纹记录于 [verification.json](evidence/phase2_20261026/verification.json)。本轮范围仅新增基线适配工具、归档文档与小型 JSON 证据；大数据、随机检查点、压缩包、vendor/workspace 均不提交 Git。
