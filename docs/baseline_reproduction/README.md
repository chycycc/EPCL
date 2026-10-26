# CASE-EPCL 外部基线复现

四模型范围已由用户确认：原版 CASE、原版 CEM、KEMP、Multi-TRS。正式训练均在与 CASE-EPCL 相同的云端环境执行。用户已回传四模型 GPU 推理通过，当前进入独立训练烟测，尚未正式训练。

- [复现方案 v1.2：范围、隔离边界与验收顺序](plans/01_classic_baselines_reproduction_plan.md)
- [源码与数据准备审计：2026-10-26](reports/01_source_and_data_preparation_audit.md)
- [统一数据与四模型 CPU 推理验收](reports/02_shared_data_and_native_inference_acceptance.md)
- [独立训练烟测入口与本地验收](reports/03_training_smoke_implementation.md)
- [云端训练烟测：增量上传与命令](../../benchmarks/classic_baselines/TRAINING_SMOKE_README.md)
- [独立工具与实际可执行的准备命令](../../benchmarks/classic_baselines/README.md)
- [固定源码来源与 SHA256](../../benchmarks/classic_baselines/sources.lock.json)
- [机器可读准备协议](../../benchmarks/classic_baselines/protocol.json)

本目录不替代 `docs/v10/` 的拆头实验计划、历史基线报告或既有选点规则。公开论文数值、作者权重补评、同环境重新训练结果须分别标注来源。近期模型属于后续候选，目前不进入四模型训练队列。
