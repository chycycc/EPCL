# 四模型源码与数据准备审计

日期：2026-10-26。范围：无 GPU 准备、官方来源、数据身份、独立工具。未启动本地/云端模型训练，未执行云端 PyTorch 2.1 前向或恢复验收。依据项目实验审计规则 v2。

## 1. 环境与工作区边界

实际分支 `v9-clean`；进入任务时规则文件、V9 文档、`docs/research/` 和多份 scratch 已有修改，均保留。没有修改 `.agents/rules/`、生产训练代码、历史测试或 sibling CEM 工程。

用户确认的云端环境是 Ubuntu 22.04 / Python 3.10 / torch 2.1.0 / CUDA 12.1 / 单卡 RTX 4090D 24GB / 18 vCPU AMD EPYC 9754。该项为用户提供的目标配置，尚非本次实测；本地工具运行在 Python 3.8 环境，不混作云端验收。

## 2. 官方源码固定

| 仓库 | 固定 commit | 受锁定的源码/文本文件数 | 用途 |
|---|---|---:|---|
| jfzhouyoo/CASE | `ca1dd9a668cc8295a3525ec75f63be194134fdc6` | 15 | 原版 CASE |
| Sahandfer/CEM | `f45d00a3cb17f48d8ff8bf39923b694c61d4695e` | 27 | 原版 CEM、CEM 作者提供的 Multi-TRS |
| qtli/KEMP | `b9a4fd58fa48556856c83d3d04125422d15e0b36` | 9 | 原版 KEMP |

来源 URL、逐文件 SHA256 在 [sources.lock.json](../../../benchmarks/classic_baselines/sources.lock.json)；实际核验见 [source_audit.json](evidence/preparation_20261026/source_audit.json)。稀疏下载只包含代码目录及根说明文件，不下载作者数据/权重，不改生产模型。

与官方 CEM 固定版本逐文件比较，忽略 CRLF/LF 后，本地不同文件为 `main.py`、`requirements.txt`、`src/models/CEM/model.py`、`src/models/common.py`。本地 Transformer/MOEL/MIME/EMPDG 源文件与该上游版本一致，但共享入口已变，所以不能仅据模型目录未变就直接复用整条训练链路。

## 3. 数据实测

CASE 与本地 CEM 的 12 个原始 `.npy` 文件 SHA256 全部相同，缓存中的情绪映射相同；**预处理 token 流不同**：

| split | 两边实际 N | context 不同的行数 | target 不同的行数 | emotion 不同的行数 |
|---|---:|---:|---:|---:|
| Train | 40,250 | 1,058 | 503 | 0 |
| Valid | 5,734 | 135 | 92 | 0 |
| Test | 5,255 | 178 | 88 | 0 |

例如 CASE 的单 token `good..` 在 CEM 中是 `good`、`..` 两个 token；这些例子仅说明编码不同，不将其简单归因为数据集不相同。完整缓存指纹、样本顺序指纹、标签计数、差异索引与目标 OOV 计数见 [data_audit.json](evidence/preparation_20261026/data_audit.json)。

实测词表为 CASE **24,647**、本地 CEM **23,712**，词表映射及 lexical-support 指纹不同。结论是旧 CEM 缓存不能直接通过统一 token 流验收，两者原生 PPL 不直接排名；不是“已经完成数据统一”。`audit_data.py` 如实返回 **1** 并保留报告，这个非零结果是预期发现，不算数据通过。

CASE 的规范样本已导出至 `results/benchmarks/classic_baselines/preparation_20261026/dataset_v2/canonical/{train,valid,test}.jsonl`。JSONL 不含模型模块引用，可作为后续适配器的逐条核验依据；不修改原缓存。KEMP 的实际知识缓存尚未核验，不填造来源或计数。

## 4. 现有源码问题与副本修补

| 项 | 固定上游位置 | 处理状态 |
|---|---|---|
| CASE/CEM 评测构造器恢复后又执行 Xavier | 两仓库 `main.py` 的 `make_model` | 新副本加 `if not is_eval`，训练初始化条件保持原样；合同测试确认评测不调用初始化 |
| 三仓库 Valid shuffle=True | 各自 data loader 的 `data_loader_val` | 新副本仅改 Valid=False，Train=True、Test=False 保留 |
| 原版 CASE `np.float` | `src/models/common.py` | 新副本改为等价 dtype `np.float64`，避免现代 NumPy 删除别名的问题 |
| KEMP seed 固定为 0 且模块重复重置 | `main.py`、`code/KEMP.py` | 待统一训练适配器显式处理；不能只伪造 `--seed 13` |
| 原生 PPL 聚合不同 | CASE/CEM 的 `common.evaluate` | 待逐样本 NLL 适配；不沿用汇总字符串混算 |
| 原生训练预算/保存时点 | 三仓库训练循环 | 仅静态记录，尚未冻结正式配方；不直接复制 EPCL 的训练步数 |
| PyTorch 2.1 / 现代依赖完整兼容 | 各模型前向、mask、优化器、解码入口 | 静态风险已列，仍需指定云端环境 smoke，不因 AST 通过而标为可训练 |

工作副本在 `benchmarks/classic_baselines/workspace/preparation_20261026_v2/`，每个模型独立。逐文件修补 diff、前后哈希和全部准备文件哈希见 [prepared_sources.json](evidence/preparation_20261026/prepared_sources.json)。`vendor` 原件未修补，没有改当前 CASE-EPCL 的 loss/PCGrad/累加/选点。

## 5. 工具验证与剩余工作

- 独立 CPU 合同测试 **12 项通过**，覆盖评测初始化保护、无执行的 AST 默认值读取、缓存全局对象限制、ID/标签错序拒绝、非有限数值拒绝、两种 PPL 聚合、micro Acc、句界 bigram、缺失生成、云端环境校验。见 [unittest.txt](evidence/preparation_20261026/unittest.txt)。这些是工具测试，不是四个模型已前向成功。
- 四个准备副本共 **70 个 Python 文件** AST 解析通过。未导入模型、不安装依赖、不执行训练。
- 源码核验命令通过；下载工具对既存副本的重复核验通过。数据命令返回 1 的分词差异完整保留。
- `score_records.py` 的合成记录 CLI 冒烟结果单独标注为合成工具测试，不进入模型结果表。
- 生产代码、历史测试、规则文件与 sibling CEM 的前后指纹核对见 [verification.json](evidence/preparation_20261026/verification.json)。

下一步仍是新缓存适配、目标编码/词表策略、KEMP 数据核验、模型各自的 artifact exporter、有效训练配方冻结，然后在指定云端环境做 smoke。当前 `cloud_ready=false`，不提供会直接误开训练的正式命令。

复核命令（从仓库根目录；数据与工作副本输出需使用新路径）：

```bash
python benchmarks/classic_baselines/fetch_sources.py
python benchmarks/classic_baselines/audit_sources.py --local-cem E:/github/CEM-master --output results/benchmarks/classic_baselines/review_source_audit.json
python benchmarks/classic_baselines/audit_data.py --case-root E:/github/CASE --cem-root E:/github/CEM-master --output-dir results/benchmarks/classic_baselines/review_dataset
python benchmarks/classic_baselines/prepare_workspaces.py --workspace benchmarks/classic_baselines/workspace/review_copy
python -m unittest discover -s benchmarks/classic_baselines/tests -v
git -c safe.directory=E:/github/CASE diff -- main.py main.sh src tests
```
