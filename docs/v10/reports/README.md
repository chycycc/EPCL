# CASE-EPCL V10 实验报告归档索引

本目录用于归档 V10 系列各 Trial 的本地工程验收报告与云端回传结果审计报告。

**讨论实验时先读：[07 实验结论与下一步决策记录](07_v10_trial1_findings_and_next_steps.md)。** 05/06保留详细统计与审计证据；07按“问题、结果、解释、决策、下一步”组织，下一轮建议尚未立项。

## 当前计划归档清单

- **V10 Trial 1 (`v10_trial1_sep_dec_head`)**
  - 事前方案（v1.2 评测与报告层修订；v1.1 训练配置及验收标准不变）：[`../plans/v10_trial1_sep_dec_head_plan.md`](../plans/v10_trial1_sep_dec_head_plan.md)
  - 工程实装与本地三阶段验收报告：[`01_v10_trial1_implementation_and_local_verification.md`](01_v10_trial1_implementation_and_local_verification.md)
  - 本地验证结构化证据：[`../../../results/v10/trial1_validation/verification.json`](../../../results/v10/trial1_validation/verification.json)
  - 云端打包待下载目录：`~/autodl-tmp/download/v10_trial1_sep_dec_head/`（及 `save/v10_trial1_sep_dec_head/CASE_*_best`）
  - 本地手动回传归档目录：`results/cloud_archive/v10_trial1_sep_dec_head/`
  - 独立复核：[`02_v10_trial1_independent_review.md`](02_v10_trial1_independent_review.md)
  - 修正与补充验收：[`03_v10_trial1_corrections_and_acceptance.md`](03_v10_trial1_corrections_and_acceptance.md)
  - v1.2 协议复核与文档补正：[`04_v10_trial1_v1_2_protocol_review.md`](04_v10_trial1_v1_2_protocol_review.md)
  - 历史参照补评、配对统计与全量生成：[`05_historical_reference_local_evaluation.md`](05_historical_reference_local_evaluation.md)
  - 历史参照原始证据：[`evidence/history_reference_20261026/verification.json`](evidence/history_reference_20261026/verification.json)
  - 可提交证据：[`evidence/trial1_correction/verification.json`](evidence/trial1_correction/verification.json)
  - 状态（2026-10-26）：**云端训练已回传；正式 Test PPL32.61 / Acc41.75% / EMO_loss2.4438，未通过冻结双门槛。V10 本地 Test/配对统计/全量生成/20点轨迹审计全部完成，主线保留T11-B。**

  - 云端回传与完整结果审计：[06_v10_trial1_cloud_results_audit.md](06_v10_trial1_cloud_results_audit.md)
  - 云端及本地复核证据：[evidence/trial1_cloud_return_20261026/verification.json](evidence/trial1_cloud_return_20261026/verification.json)
