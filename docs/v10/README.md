# CASE-EPCL V10 实验与文档导航手册

> **项目标识**：CASE-EPCL (V10 现有组件配合优化与多任务训练解耦)
>
> **对照基线**：**V9 T11-B** (`v9_trial11_B_gnorm6`, Test $N=5255$/`batch=1` 样本 macro PPL **32.56** / 样本 micro Acc **42.21%** / EMO_loss **2.2840**)
>
> **研究定位**：停止继续堆叠新模块，转向现有已启用组件之间的梯度、表征与损失调度配合优化；目录整理与模型改动严格分开处理。
>
> **优先阅读**：[Trial1 实验结论与下一步决策记录](reports/07_v10_trial1_findings_and_next_steps.md)——集中说明结果、可能原因、主线决策与下一轮建议。详细验收证据留在05/06报告。
>
> **当前状态（2026-10-26）**：用户已完成 **V10 Trial 1** 云端训练并回传归档。正式 Test `32.61 / 41.75% / 2.4438` 未通过冻结双门槛，主线保留 T11-B。历史参照补评与全量生成已完成，见[报告05](reports/05_historical_reference_local_evaluation.md)；V10 本地两份检查点 Test、配对统计、全量5255生成与云端轨迹审计全部完成，见[报告06](reports/06_v10_trial1_cloud_results_audit.md)。事前方案仍为[v1.2](plans/v10_trial1_sep_dec_head_plan.md)，训练代码与判据不变。

---

## 一、目录结构

```text
docs/v10/
├── README.md                                    # [本文件] V10 定位说明与导航索引
├── v10_experiment_log.md                        # V10 实验总账（锚定 V9 T11-B 基线）
├── plans/                                       # 事前预注册方案目录
│   └── v10_trial1_sep_dec_head_plan.md          # v1.2 评测与报告层修订；训练配置不变
└── reports/                                     # 评测与因果审计报告目录
    ├── README.md                                # 报告归档索引
    ├── 01_v10_trial1_implementation_and_local_verification.md
    ├── 02_v10_trial1_independent_review.md       # 修正前独立复核
    ├── 03_v10_trial1_corrections_and_acceptance.md
    ├── 04_v10_trial1_v1_2_protocol_review.md
    ├── 05_historical_reference_local_evaluation.md
    ├── 06_v10_trial1_cloud_results_audit.md
    ├── 07_v10_trial1_findings_and_next_steps.md    # 结果、解释与下一步建议
    └── evidence/                              # 工程验收、历史参照和云端回传的小型原始证据
```

对应端云回传路径分工：
- **云端打包输出目录（`main.sh` 自动汇总）**：`${HOME}/autodl-tmp/download/v10_trial1_sep_dec_head/`（权重位于云端 `save/v10_trial1_sep_dec_head/`）
- **本地回传归档目录（手动下载后存放）**：`results/cloud_archive/v10_trial1_sep_dec_head/`

---

## 二、核心准则与与 V9 的衔接

1. **V9 收尾状态**：
   - **V9 T11-B**（[`results/cloud_archive/v9_trial11_B_gnorm6/`](../../results/cloud_archive/v9_trial11_B_gnorm6/)）保留为主线检查点；
   - V9 Trial 12（四质心自注意力前缀路线）已于 2026-10-24 正式封盘停止（见 [`docs/v9/reports/trial12/08_prefix_norm_results_audit.md`](../v9/reports/trial12/08_prefix_norm_results_audit.md)），代码开关默认 `False`，产物与审计保留备查。
2. **评价标准不重置**：
   - 主线替换门槛继续锚定 **V9 T11-B**：正式 Test（$N=5255$、`batch=1`、`shuffle=False`）**样本 macro PPL ≤ 32.56 且 样本 micro EMO_acc ≥ 42.21%**；
   - T10、T11（seed13）列为必须报告的历史参照；T11-B 保留为唯一直接对照，不新增训练组、不切换底座。
   - 生成主口径为本地同环境 Test 全量 $N=5255$，云端顺序前 $N=500$ 保留为历史连续性列，两者不可混比；均为 `batch=1`、`max_dec_step=40`，并列报告 `Greedy Dist-2`、`Greedy Unique`、`Beam Dist-2`（附 `Beam Unique`）。T11 本地无权重，全量列标未测。
3. **方法论与工程纪律**：
   - 严格遵守 [`.agents/rules/case-epcl-experiment-audit-rules.md`](../../.agents/rules/case-epcl-experiment-audit-rules.md) v2 的五条审计铁律；
   - 统一通过 [`main.sh`](../../main.sh) 启动训练；
   - 目录整理与模型改动分开处理，不在实验轮次中顺带删除旧组件或重构训练逻辑。
