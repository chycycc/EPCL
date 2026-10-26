# 四模型基线复现准备工具

本目录独立于 CASE-EPCL 生产训练链路。范围为原版 CASE、原版 CEM、KEMP、Multi-TRS；云端目标配置见 [cloud_target.json](cloud_target.json)，准备协议见 [protocol.json](protocol.json)。正式训练配方尚未冻结，`cloud_ready=false`。

## 目录边界

- `vendor/`：按 [sources.lock.json](sources.lock.json) 固定 commit 的官方源码，已忽略；只读核验，不放模型权重或数据。
- `workspace/`：从官方源码复制的独立工作副本，已忽略；每个模型有自己的顶层 `src` 或 `code`、配置和 `_baseline_provenance.json`。
- `audit_sources.py`：源码、文件 SHA256、静态 CLI 默认值和本地 CEM 差异核验，不导入模型。
- `audit_data.py`：读取已知本地缓存，导出普通 JSONL 样本清单并逐条比较；不执行原 data loader，不写原缓存。
- `prepare_workspaces.py`：只修改新副本，记录逐文件 diff/哈希；目前包括 Valid 不打乱、CASE/CEM 测试恢复后跳过 Xavier、CASE 的 `np.float` 等价替换为 `np.float64`。
- `score_records.py`：对未来适配器导出的 JSONL 计算指标，校验 ID/标签/数值完整性。它不负责运行模型或加载权重。
- `cloud_preflight.py`：在云端显式运行，记录并校验环境，不训练。只有这个命令会导入 torch。
- `tests/`：本套件的 CPU 合同测试，与根目录历史测试分开运行。

所有模型都应在其工作副本的独立子进程和工作目录中运行。不通过 `sys.path` 将多个上游工程同时导入一个进程，不导入 CASE-EPCL 的 `src`，不调用主项目的自动清理评测入口。当前工具不提供训练子命令；数据和运行时检查通过后再补专用适配器。

## 准备命令

从 CASE 仓库根目录运行，使用当前 Python。以下都是准备命令，不会训练：

```bash
python benchmarks/classic_baselines/fetch_sources.py
python benchmarks/classic_baselines/audit_sources.py --output results/benchmarks/classic_baselines/source_audit.json
python benchmarks/classic_baselines/prepare_workspaces.py --workspace benchmarks/classic_baselines/workspace/new_preparation
python -m unittest discover -s benchmarks/classic_baselines/tests -v
```

`fetch_sources.py` 只取三份官方仓库的源码；Multi-TRS 和 CEM 共享官方代码来源，但工作副本独立。现存 vendor 目录版本或文件不匹配时直接失败，不 reset。`prepare_workspaces.py` 拒绝覆盖既存工作目录。第三方源码的授权与引用信息保留在各自副本内。

本地实际数据检查命令（必须使用新输出目录）：

```bash
python benchmarks/classic_baselines/audit_data.py --case-root E:/github/CASE --cem-root E:/github/CEM-master --output-dir results/benchmarks/classic_baselines/new_dataset_audit
```

旧 CEM 缓存分词不同，此命令按设计返回退出码 1，并保留 `data_audit.json` 与 CASE 的 `canonical/{train,valid,test}.jsonl`。这是缓存不能直接混用的检查结果。只比较样本身份必需的 context/target/emotion；模型特有知识特征、situation、说话人角色及知识构造来源仍由后续适配器单独核验。

## 云端环境核验

先在 CASE-EPCL 实际环境中生成参考，再在每个基线所用环境核验；不要在本地旧 Python 中把它标为云端验收。

```bash
python benchmarks/classic_baselines/cloud_preflight.py --output results/benchmarks/classic_baselines/case_epcl_cloud_environment.json
python benchmarks/classic_baselines/cloud_preflight.py --reference results/benchmarks/classic_baselines/case_epcl_cloud_environment.json --output results/benchmarks/classic_baselines/baseline_cloud_environment.json
```

检查 Ubuntu 22.04 / Python 3.10 / PyTorch 2.1.0 / torch CUDA 12.1 / 单张 4090D；对比完整 Python、torch、cuDNN 和相关包版本。CPU 型号、affinity、cgroup quota 留存，18 vCPU 分配仍以实例实际配置核验。不要为迁就旧 requirements.txt 降级或覆盖 CASE-EPCL 环境；兼容修补先在工作副本做 diff 记录。

## 指标文件合同

适配器逐样本输出 `sample_id`（来自 canonical JSONL）、`emotion_gold`（名称）、`nll_sum`、`token_count`。NLL 是目标 token 的未平滑生成负对数似然之和，排除 PAD、包含 EOS，不能混入情绪、BOW、MIM、对比损失；pointer 模型必须声明实际概率支持。分类模型另输出 `emotion_pred` 名称与未加权 `emotion_ce`。生成输出统一文本规范化后的 `greedy_tokens`、`beam_tokens`，均为 token 字符串数组。

元数据必需字段：`model`、`checkpoint_sha256`、`source_commit`、`environment_sha256`、`target_encoding_sha256`、`canonical_sha256`、`nll_definition`、`text_normalization`、`generation_protocol`。NLL 定义的精确字符串见协议 JSON。适配器必须先核验来源；本工具检查格式和 canonical 文件哈希，不替代模型恢复审计。

```bash
python benchmarks/classic_baselines/score_records.py --records /run/test_records.jsonl --canonical /data/canonical/test.jsonl --metadata /run/metadata.json --output /run/metrics.json
```

输出样本 macro PPL、token micro PPL、分类 micro Acc/样本 CE，及 Greedy/Beam 的全量与前 500 指标。样本数按实际输入计算；smoke 子集不能被称为全量 Test。不同 target encoding 指纹或概率支持的 PPL 不直接排名。完整复现方案和当前证据见 [docs/baseline_reproduction](../../docs/baseline_reproduction/README.md)。
