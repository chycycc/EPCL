# V11 方案 B（PCAM 内部 LayerNorm 重构）与 PCMG 改进筛选报告

## 1. 历史消融核查与审稿红线确认

| 审计主题 | 事实与数值证据 | 出处 |
| :--- | :--- | :--- |
| **`hop=1` 审稿公平性红线** | `benchmarks/classic_baselines/` 中全部 6 个经典基线（`Transformer, MIME, CEM, MoEL, CASE, EMPDG`）默认编码器与解码器层数**严格为 `hop=1`**。主实验表必须严守 `hop=1`，`dec_layers=2`（D4）仅可作为论文附加 Scaling 分析，避免审稿人质疑主表收益来自解码器加倍。 | [unified_case_evaluation_comparison.md#L21-L63](file:///docs/research/unified_case_evaluation_comparison.md#L21-L63) |
| **PCAM 从头训练消融（V9 T11 vs T11-A）** | 从头训练移除 PCAM（`w/o PCAM`，减少 `542,101` 参数）导致 **Test PPL 从 `32.62` 恶化至 `33.63 (+1.01)`**，**Greedy Dist-2 从 `18.94%` 塌陷至 `15.40% (-3.54pp)`**，**Beam Dist-2 从 `13.16%` 跌至 `10.83% (-2.33pp)`**。PCAM 是生成与多样性的核心支柱。 | [v9_trial11_A_cross_audit.md#L69-L89](file:///docs/v9/reports/v9_trial11_A_cross_audit.md#L69-L89) |
| **PCMG 与 `use_masked_react_pool` 贡献拆解（V10）** | 1. **`use_pcmg_head`（T7 vs T2）**：Test Acc `41.83% → 42.36% (+0.53pp)`，EMO_loss `2.3545 → 2.2623 (-0.0922)`，但 **Test PPL 从 `32.29` 退化至 `32.46 (+0.17)`**。<br>2. **`use_masked_react_pool`**：在 T7 冻结权重上 Test Acc 增益为 **`0.00pp`**（`2226/5255` 不变），EMO_loss `-0.0022`。<br>3. **T9 词汇专家（`LinearSVC`）**：贡献了 `+2.38pp` Acc（`42.36% → 44.74%`）与 `-0.2509` EMO_loss。 | [v10_trial7_audit.md#L16-L20](file:///docs/v10/reports/v10_trial7_audit.md#L16-L20)<br>[v10_experiment_log.md#L220-L347](file:///docs/v10/v10_experiment_log.md#L220-L347) |

---

## 2. `hop=1` 零新增参数候选全集 1,500 Updates 终局横向对比（含叠加验证）

所有实验臂严格保持 **`hop=1`、独立去重参数量 `30,336,257`（零参数增量）、相同随机种子与确定性批次序列**（`bsz=4, accum=8`，全量 Valid `N=5734` TF + `N=256` Greedy）：

| 实验臂与开关配置 | twPPL ↓ | bPPL ↓ | EMO_loss ↓ | Acc (%) ↑ | dec_emo_acc (%) ↑ | D1 (%) ↑ | D2 (%) ↑ | 综合裁决 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Base_scratch`**（V10 T7+masked_pool 基准） | 51.86 | 48.14 | 2.0726 | 40.41% | 39.43% | 4.75% | 12.28% | 共同基准 |
| **`Cand_optB`**（方案 B：`--pcam_inner_ln`） | **51.04 (`-0.82`)** | **47.34 (`-0.80`)** | **2.0227 (`-0.050`)** | **42.15% (`+1.74`)** | **40.11% (`+0.68`)** | **4.79% (`+0.04`)** | 11.65% (`-0.62`) | **全面胜出（唯一保留主线）** |
| **`Cand_pcmg1`**（`--pcmg_detach_proto_cal`） | 51.04 (`-0.82`) | 47.37 (`-0.77`) | 2.0717 (`-0.001`) | 40.91% (`+0.51`) | 39.20% (`-0.23`) | 4.55% (`-0.20`) | 11.10% (`-1.17`) | 收益被 `Cand_optB` 覆盖（淘汰） |
| **`Cand_optB_pcmg1`**（`optB + pcmg1` 叠加） | 50.97 (`-0.89`) | 47.28 (`-0.86`) | 2.0349 (`-0.038`) | 41.58% (`+1.17`) | 38.98% (`-0.45`) | 4.83% (`+0.09`) | 11.55% (`-0.73`) | **相对 `Cand_optB` 仅 `-0.08` PPL，但 Acc `-0.58pp`、`dec_emo_acc -1.13pp`（淘汰叠加）** |
| **`Cand_pcmg2`**（`--pcmg_proto_salience_pool`） | 51.86 (`-0.00`) | 48.12 (`-0.02`) | 2.0564 (`-0.016`) | 41.51% (`+1.10`) | 39.52% (`+0.09`) | 4.31% (`-0.44`) | 9.88% (`-2.40`) | PPL 零收益且 D2 退化（淘汰） |
| **`Cand_pcmg3`**（`--pcmg_dec_proto_cal`） | 50.98 (`-0.88`) | 47.29 (`-0.85`) | 2.0638 (`-0.009`) | 40.77% (`+0.37`) | 37.57% (`-1.87`) | 4.96% (`+0.21`) | 11.75% (`-0.52`) | `dec_emo_acc` 严重退化 `-1.87pp`（淘汰） |

---

## 3. 方案 B（`Cand_optB`：`--pcam_inner_ln`）与 `Cand_optB_pcmg1` 叠加验证全轨迹

数据源：[result_Base_scratch.json](file:///scratch/t11_weight_tying/result_Base_scratch.json)、[result_Cand_optB.json](file:///scratch/t11_weight_tying/result_Cand_optB.json)、[result_Cand_pcmg1.json](file:///scratch/t11_weight_tying/result_Cand_pcmg1.json)、[result_Cand_optB_pcmg1.json](file:///scratch/t11_weight_tying/result_Cand_optB_pcmg1.json)。

| 检查点 | 指标 | Base_scratch | Cand_optB | Cand_pcmg1 | Cand_optB_pcmg1 | 叠加相对 OptB 净增量 (`optB_pcmg1 - optB`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **500** | **twPPL / bPPL** | 72.48 / 67.34 | 71.58 / 66.47 | 70.90 / 65.84 | 70.74 / 65.66 | `-0.83 / -0.81` |
| | **EMO_loss** | 2.2049 | 2.1927 | 2.1626 | 2.1972 | `+0.0044` |
| | **Acc / dec_emo_acc (%)** | 36.31 / 31.17 | 37.25 / 29.82 | 37.46 / 30.57 | 36.68 / 30.10 | `-0.58 / +0.28` |
| **1000** | **twPPL / bPPL** | 57.14 / 53.33 | 55.88 / 52.15 | 55.98 / 52.25 | 55.85 / 52.10 | `-0.03 / -0.05` |
| | **EMO_loss** | 2.1369 | 2.0981 | 2.0875 | 2.0823 | `-0.0157` |
| | **Acc / dec_emo_acc (%)** | 38.19 / 35.04 | 39.20 / 34.71 | 39.17 / 35.12 | 39.08 / 35.39 | `-0.12 / +0.68` |
| **1500** | **twPPL / bPPL** | 51.86 / 48.14 | **51.04 / 47.34** | 51.04 / 47.37 | 50.97 / 47.28 | **`-0.08 / -0.06`（收益高度重合）** |
| | **EMO_loss** | 2.0726 | **2.0227** | 2.0717 | 2.0349 | **`+0.0122`（劣化）** |
| | **Acc / dec_emo_acc (%)** | 40.41 / 39.43 | **42.15 / 40.11** | 40.91 / 39.20 | 41.58 / 38.98 | **`-0.58 / -1.13`（双分类准确率退化）** |
| | **D1 / D2 (%)** | 4.75 / 12.28 | **4.79 / 11.65** | 4.55 / 11.10 | 4.83 / 11.55 | `+0.05 / -0.11` |

### 叠加实验结论（回答上云策略）
- 在 `Cand_optB` 基础上叠加 `--pcmg_detach_proto_cal`（`Cand_optB_pcmg1`），1,500 步 `twPPL` 仅微降 `-0.0759`（说明两者的 PPL 改善有 `91%` 以上重合），同时导致主分类 `Acc` 回吐 `-0.58pp`、解码端 `dec_emo_acc` 回吐 `-1.13pp`、`EMO_loss` 上升 `+0.0122`。
- **云端决策**：**彻底淘汰 `Cand_pcmg1` 及叠加方案 `Cand_optB_pcmg1`**。上云时无需跑 `Cand_pcmg1` 或叠加版，直接单跑 `v11_optB_pcam_inner_ln`（纯方案 B），节省 2 次云端全量训练。

---

## 4. PCMG 三项单变量改进（`Cand_pcmg1 / 2 / 3`）全轨迹详情

数据源：[result_Cand_pcmg1.json](file:///scratch/t11_weight_tying/result_Cand_pcmg1.json)、[result_Cand_pcmg2.json](file:///scratch/t11_weight_tying/result_Cand_pcmg2.json)、[result_Cand_pcmg3.json](file:///scratch/t11_weight_tying/result_Cand_pcmg3.json)。

| 检查点 | 指标 | Base_scratch | Cand_pcmg1 (`detach_proto_cal`) | Cand_pcmg2 (`proto_salience_pool`) | Cand_pcmg3 (`dec_proto_cal`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **500** | **twPPL / bPPL** | 72.48 / 67.34 | 70.90 / 65.84 | 71.51 / 66.47 | 70.24 / 65.24 |
| | **EMO_loss** | 2.2049 | 2.1626 (`-0.0423`) | 2.2017 (`-0.0032`) | 2.2102 (`+0.0053`) |
| | **Acc / dec_emo_acc (%)** | 36.31 / 31.17 | 37.46 / 30.57 | 36.71 / 29.42 | 36.80 / 28.93 |
| **1000** | **twPPL / bPPL** | 57.14 / 53.33 | 55.98 / 52.25 | 56.54 / 52.77 | 55.40 / 51.61 |
| | **EMO_loss** | 2.1369 | 2.0875 (`-0.0495`) | 2.1123 (`-0.0247`) | 2.1129 (`-0.0241`) |
| | **Acc / dec_emo_acc (%)** | 38.19 / 35.04 | 39.17 / 35.12 | 38.59 / 35.89 | 38.28 / 34.60 |
| **1500** | **twPPL / bPPL** | 51.86 / 48.14 | 51.04 / 47.37 | 51.86 / 48.12 | 50.98 / 47.29 |
| | **EMO_loss** | 2.0726 | 2.0717 (`-0.0008`) | 2.0564 (`-0.0162`) | 2.0638 (`-0.0088`) |
| | **Acc / dec_emo_acc (%)** | 40.41 / 39.43 | 40.91 / 39.20 | 41.51 / 39.52 | 40.77 / 37.57 |
| | **D1 / D2 (%)** | 4.75 / 12.28 | 4.55 / 11.10 | 4.31 / 9.88 | 4.96 / 11.75 |
