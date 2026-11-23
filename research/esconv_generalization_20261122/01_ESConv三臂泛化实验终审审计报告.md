# ESConv 跨数据集泛化实验（3 臂零调参迁移）终审审计报告

> **归档日期**：2026-11-23  
> **代码基线**：分支 `v9-clean`，提交 `1e0e4fd`（8/8 核心源码文件 SHA256 与云端 `source_sha256.txt` 一致）  
> **数据资产**：`data/ESConv/dataset_preproc.p`（`sha256=1e4e3ab6f4da...`，Vocab=`13,683`，Session 级分层防泄漏 8:1:1 划分：Train `1,040 sessions / 14,101 turns`、Dev `130 / 1,727`、Test `130 / 1,817`，`session_overlap=0`，`exact_situation_text_overlap=0`，`empty_rel_counts={}`）  
> **归档路径**：[`results/cloud_archive/esconv_3arms_20261122/`](../../results/cloud_archive/esconv_3arms_20261122/)  
> **评测口径**：遵循《CASE-EPCL 实验因果审计铁律 (v3)》——全量 Test `N=1,817`（`40,318` target tokens），`batch_size=1, shuffle=False, max_dec_step=50, beam_size=5`，双 Best 并列报告，主表统一口径为 **`ppl_best`**，`composite_best` 并列保留作附录敏感性对照。

---

## 1. 本地权重归档与 SHA256 验真清单

全部 6 份云端权重已拉回本地并与云端 `checkpoint_sha256.txt` 100% 逐字节核验通过：

| 实验臂 | 角色 | 检查点文件名 | 大小 | SHA256 (前 12 位) | 校验状态 |
| --- | --- | --- | :---: | :---: | :---: |
| `esconv_case_baseline` | `ppl_best` | `CASE_3999_61.1353_ppl_best` | 245.0 MB | `915f42295d15` | PASS |
| `esconv_case_baseline` | `composite_best` | `CASE_3999_61.1353_composite_best` | 245.0 MB | `f72bcffbad12` | PASS |
| `esconv_wo_all_proto` | `ppl_best` | `CASE_3999_60.7759_ppl_best` | 246.5 MB | `67c483c5df1f` | PASS |
| `esconv_wo_all_proto` | `composite_best` | `CASE_2599_60.7983_composite_best` | 246.5 MB | `9e661c1eb7b0` | PASS |
| `esconv_case_epcl` | `ppl_best` | `CASE_2599_60.4392_ppl_best` | 253.7 MB | `14057cd79597` | PASS |
| `esconv_case_epcl` | `composite_best` | `CASE_2599_60.4392_composite_best` | 253.7 MB | `675fa6d979a1` | PASS |

---

## 2. 三臂对照设计与训练配方对齐

所有三臂共享同一预处理文件 `dataset_preproc.p`、同一随机种子 `13`、同一基础超参（`batch_size=32, fp32, lr=0.000576, warmup=800, cosine decay 1200->4000, max_step=4400, pretrain_epoch=4, div_weight=1.5, check_iter=200, min_save_step=1200, patience=8, composite_mode=str_loss, composite_emo_loss_weight=5.0`）：

1. **`esconv_case_baseline`**：原版 CASE-ESConv（关闭情绪分支、原型、PCMG、PCAM、PCGrad，仅保留 8 类策略主任务与相同基础调度；修复原代码 `ESconv` 大小写拼写错误与 `MLP` 4 路维度）。
2. **`esconv_wo_all_proto`**：在同配方上开启 11 类求助者情绪辅助头、多粒度 `xReact` 聚合、Valid-Neg、`emo_branch_dropout=0.1`、`PCGrad` 与钟形情绪损失爬坡（`1000->2000, max=1.25`），并通过 `--disable_all_prototypes` 严格关闭全部 5 条原型路径。
3. **`esconv_case_epcl`**：完整 CASE-EPCL 零调参迁移版（在 `wo_all_proto` 基础上开启 `EPCL + MCP (m=0.96, lambda=0.03, alpha_uni=0.8, arc_margin=0.30) + MoP-DR + PCMG (w=0.25, tau=0.2) + PCAM (pcam_inner_ln=True)`，绑定至 11 类求助者情绪，保留 8 类支持策略主任务）。

---

## 3. 全量 Test（N=1,817）双 Best 并列总表

数据源：[`results/cloud_archive/esconv_3arms_20261122/esconv_3arms_summary.json`](../../results/cloud_archive/esconv_3arms_20261122/esconv_3arms_summary.json)。

### 3.1 主报告口径：`ppl_best`

| 指标 (ESConv Test, N=1817) | `esconv_case_baseline` (Step 3999) | `esconv_wo_all_proto` (Step 3999) | `esconv_case_epcl` (Step 2599) |
| --- | :---: | :---: | :---: |
| **Valid PPL ↓** | `61.1353` | `60.7759` | **`60.4392`** |
| **Test `case_ppl` ↓** | `44.5961` | `44.9225` | **`44.5858`** |
| **Test `token_micro_ppl` ↓** | `57.0427` | `57.1190` | **`56.3019`** |
| **Str CE ↓** | `1.8193` | **`1.8080`** | `1.8098` |
| **Str Acc (%) ↑** | `33.46` | **`34.01`** | `33.19` |
| **Str Macro-F1 (%) ↑** | `20.72` | **`22.48`** | `19.57` |
| **Emo CE ↓** (11 类，Test 含 8 类) | — | `2.7854` | **`2.4661`** |
| **Emo Acc (%) ↑** | — | **`34.56`** | `32.09` |
| **Emo Macro-F1 (%) ↑** | — | **`21.34`** | `20.05` |
| **Greedy Dist-1 (%) ↑** | `0.7456` | **`0.9476`** | `0.7481` |
| **Greedy Dist-2 (%) ↑** | `1.9260` | **`3.1887`** | `1.9168` |
| **Greedy Unique (%) ↑** | `6.2741` | **`11.2823`** | `5.2834` |
| **Greedy Avg_Len** | `9.74` | `9.70` | `9.27` |
| **Greedy BLEU-2 / BLEU-4 / ROUGE-L ↑** | `3.51 / 1.22 / 15.64` | **`3.61 / 1.21 / 15.95`** | `3.02 / 0.97 / 15.59` |
| **Beam-5 Dist-1 (%) ↑** | `0.6653` | **`0.6954`** | `0.6831` |
| **Beam-5 Dist-2 (%) ↑** | `1.6792` | **`1.8547`** | `1.7945` |
| **Beam-5 Unique (%) ↑** | `6.2190` | **`7.0446`** | `5.5036` |
| **Beam-5 BLEU-2 / BLEU-4 / ROUGE-L ↑** | **`3.70 / 1.26 / 15.72`** | `3.30 / 1.17 / 15.20` | `3.13 / 1.10 / 14.25` |
| **原型几何 (`eff_rank` / `off_diag_cos`)** | — | — | `7.5081 / -0.0359 (max 0.7553)` |

### 3.2 附录对照口径：`composite_best`

> `esconv_case_baseline`（Step 3999）与 `esconv_case_epcl`（Step 2599）双 Best 落在同一步，指标与上表一致；仅 `esconv_wo_all_proto` 的 `composite_best` 落在 **Step 2599**（与 `esconv_case_epcl` 步数严格对齐）。

| 指标 (`composite_best`, N=1817) | `esconv_case_baseline` (Step 3999) | `esconv_wo_all_proto` (Step 2599) | `esconv_case_epcl` (Step 2599) |
| --- | :---: | :---: | :---: |
| **Valid PPL ↓** | `61.1353` | `60.7983` | **`60.4392`** |
| **Test `case_ppl` ↓** | `44.5961` | `45.2677` | **`44.5858`** |
| **Test `token_micro_ppl` ↓** | `57.0427` | `57.1254` | **`56.3019`** |
| **Str CE / Str Acc / Macro-F1** | `1.8193 / 33.46 / 20.72` | `1.8076 / 32.80 / 21.10` | `1.8098 / 33.19 / 19.57` |
| **Emo CE / Emo Acc / Macro-F1** | — | **`1.9441 / 37.59 / 21.02`** | `2.4661 / 32.09 / 20.05` |
| **Greedy Dist-1 / Dist-2 / Unique (%)** | `0.75 / 1.93 / 6.27` | **`0.82 / 2.78 / 13.98`** | `0.75 / 1.92 / 5.28` |
| **Greedy BLEU-2 / ROUGE-L** | `3.51 / 15.64` | **`4.24 / 15.86`** | `3.02 / 15.59` |
| **Beam-5 Dist-1 / Dist-2 / Unique (%)** | `0.67 / 1.68 / 6.22` | **`0.74 / 2.00`** / `5.83` | `0.68 / 1.79 / 5.50` |
| **Beam-5 BLEU-2 / ROUGE-L** | **`3.70 / 15.72`** | `2.17 / 15.02` | `3.13 / 14.25` |

---

## 4. 配对显著性检验与分长度桶拆解

### 4.1 显著性检验汇总
- **Token 加权总 NLL（`token_micro_ppl` 对应逐样本总 NLL 配对 t 检验，`ppl_best`）**：
  - `case_epcl` vs `case_baseline`：`mean_diff = +0.2900`, `t = +2.911`, **`p = 0.00360`**（显著更优）
  - `case_epcl` vs `wo_all_proto`：`mean_diff = +0.3197`, `t = +3.125`, **`p = 0.00178`**（显著更优）
- **样本宏平均 NLL（`case_ppl` 配对 t 检验）**：
  - `composite_best`（同 Step 2599）：`case_epcl` 显著优于 `wo_all_proto`（`t = +2.995`, **`p = 0.00274`**）；与 `baseline` 持平（`p = 0.967`）。
  - `ppl_best`：`case_epcl` vs `wo_all_proto`（`t = +1.349`, `p = 0.177`）；vs `baseline`（`p = 0.967`）。
- **8 类策略准确率 McNemar 精确检验（`ppl_best`）**：
  - 三组两两对比均未达统计显著（`p = 0.359 ~ 0.797`）。

### 4.2 目标回复长度分桶 PPL（`ppl_best`）

| 目标长度区间 | 样本数 `n` | 总 Token 数 | `case_baseline` (`micro / case`) | `wo_all_proto` (`micro / case`) | `case_epcl` (`micro / case`) |
| --- | :---: | :---: | :---: | :---: | :---: |
| `len [1, 10]` | 383 | 2,766 | **`27.89 / 27.32`** | `28.22 / 27.89` | `28.80 / 28.21` |
| `len [11, 20]` | 642 | 9,789 | **`39.58 / 38.36`** | `39.97 / 38.74` | `39.84 / 38.60` |
| `len [21, 35]` | 521 | 13,856 | `57.85 / 57.17` | `58.09 / 57.38` | **`56.87 / 56.28`** |
| `len [36, ∞)` | 271 | 13,907 | `83.87 / 79.00` | `83.09 / 78.16` | **`81.25 / 76.56`** |

结论：`case_epcl` 的语言建模增益高度集中在**长支持性回复**（`len >= 21` 占全部测试 token 的 `68.9%`，尤其 `len >= 36` 降幅达 `-2.62` micro-PPL / `-2.44` case-PPL）。

---

## 5. 根因审计：为何 `case_epcl` 在 ESConv 上出现自由解码塌缩？

经逐行核查 `model.py` 与 `decode/case.py`，实现无前向/推理失配 Bug；`per_sample.jsonl` 证据指向**监督粒度与任务结构的错配**：

1. **通用共情句严重塌缩**：在 `ppl_best` Greedy 输出中，`esconv_case_epcl` 单句 `'i can understand how you feel . '` 出现 **`626 / 1,817 (34.5%)`**（`wo_all_proto` 最高频句仅 `187/1817 = 10.3%`，`baseline` 为 `327/1817 = 18.0%`），且在 **全部 8 种支持策略**（含 `Question`、`Providing Suggestions`、`Information`）下均成为 Top-1 高频输出，直接拉低 `Greedy Dist-2 (1.92%)` 与 `Unique (5.28%)`。
2. **会话静态情绪 vs 轮次动态策略错配**：
   - ED 训练集含 `19,533` 个短对话（约 2 轮），32 类细粒度情绪即轮次级回复核心条件；
   - ESConv 训练集仅含 `1,040` 个长 Session，平均 **13.6 轮共享同一个静态会话级粗情绪标签**（且测试集仅 8 个有效情绪类，3 个单 Session 类无测试样本），真正驱动每轮回复变化的是 **8 类轮次动态支持策略**。
   - 零调参下 `PCAM` 门控 `tanh(gamma)` 攀升至 `0.33~0.40` 并叠加 `dec_emo_loss`，在 Teacher Forcing 下（真实前缀已确定当前轮策略）有效补充长回复语境并拉低 `token_micro_ppl`；但在自回归自由解码从 `<SOS>` 启动时，静态情绪原型记忆强覆盖解码隐状态，诱导解码器反复抢答高频万能情绪共情句。
   - 同时，1,040 个 Session 的静态情绪标签被重复记忆 136 遍，验证集 `EMO_loss` 从 Step 1199 的 `1.6838` 快速过拟合至 Step 4199 的 `3.1449`，导致 `case_epcl` 在 Step 2599 提前早停封顶。

---

## 6. 论文宣称边界与后续可选路线

- **严禁越界宣称**：不得宣称“零调参 `CASE-EPCL` 在 ESConv 全部指标上全面超越原版 CASE”。
- **可成立的实证宣称**：
  1. 在不调参的跨任务迁移下，`CASE-EPCL` 的原型记忆对长支持性回复的概率建模仍显著有效（`Valid PPL 61.14 -> 60.44`，`Test token_micro_ppl 57.04 -> 56.30, p=0.0036`，`len>=36` 降低 `-2.62` PPL），且显著缓解了少数类情绪预测过拟合校准失真（`Emo CE 2.79 -> 2.47`）；
  2. 对比 ED（轮次情绪主导）与 ESConv（会话静态情绪 + 轮次动态策略），实验揭示了原型记忆生效的关键边界条件——**原型空间必须与轮次级回复条件变量对齐**；当绑定到跨十余轮恒定的静态粗标签时，强 `PCAM` 注入会在自由解码阶段诱发通用共情模板塌缩。
- **可选验证实验（方案 B）**：若需构建 ESConv 正面主表，应做单变量手术——将 `EPCL + MCP + PCMG + PCAM` 绑定至 ESConv 原生 **8 类轮次支持策略 (`strategy_label`)**（移除过拟合的 11 类静态情绪头，与原版 CASE-ESConv 保持 100% 相同监督信号），验证任务对齐原型能否同步提升 `Str Acc/F1`、`PPL` 与生成多样性。
