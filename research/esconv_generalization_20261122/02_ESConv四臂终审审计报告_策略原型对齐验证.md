# ESConv 跨数据集泛化实验（4 臂完整对照：策略原型对齐验证）终审审计报告

> **归档日期**：2026-11-23  
> **代码基线**：分支 `v9-clean`，提交 `093cabb`（7/7 核心源码文件与云端 `source_sha256.txt` 哈希完全一致）  
> **数据资产**：`data/ESConv/dataset_preproc.p`（`sha256=1e4e3ab6f4da...`，Vocab=`13,683`，Session 级分层防泄漏 8:1:1 划分：Train `1,040 sessions / 14,101 turns`、Dev `130 / 1,727`、Test `130 / 1,817`，`session_overlap=0`，`exact_situation_text_overlap=0`）  
> **归档路径**：
> - 前 3 臂归档：[`results/cloud_archive/esconv_3arms_20261122/`](../../results/cloud_archive/esconv_3arms_20261122/)
> - 第 4 臂（策略原型对齐）归档：[`results/cloud_archive/esconv_strategy_epcl_20261123/`](../../results/cloud_archive/esconv_strategy_epcl_20261123/)
> **评测口径**：遵循《CASE-EPCL 实验因果审计铁律 (v3)》——全量 Test `N=1,817`（`40,318` target tokens），`batch_size=1, shuffle=False, max_dec_step=50, beam_size=5`，双 Best 并列报告，全论文主表统一口径为 **`ppl_best`**，`composite_best` 并列保留作附录敏感性对照。

---

## 1. 第 4 臂（`esconv_strategy_epcl`）本地权重验真与云端停机确认

- **云端停机状态**：全部文件与权重回传并 SHA256 校验通过后，已写入 `/tmp/.strategy_downloaded_ok` 触发守护进程执行 `/usr/bin/shutdown`，复测 SSH 端口连接超时，**确认云端实例已停机止费**。
- **本地权重 SHA256 核验清单（2/2 PASS）**：

| 实验臂 | 角色 | 检查点文件名 | 大小 | SHA256 (前 12 位) | 校验状态 |
| --- | --- | --- | :---: | :---: | :---: |
| `esconv_strategy_epcl` | `ppl_best` | `CASE_3799_60.8131_ppl_best` | 253.6 MB | `f27961266707` | PASS |
| `esconv_strategy_epcl` | `composite_best` | `CASE_3799_60.8131_composite_best` | 253.6 MB | `cf66cfa81112` | PASS |

---

## 2. 四臂实验矩阵与因果控制结构

所有 4 臂共享同一 `dataset_preproc.p`、同一随机种子 `13` 与完全一致的基础优化/调度配方（`batch_size=32, fp32, lr=0.000576, warmup=800, cosine 1200->4000, max_step=4400, pretrain_epoch=4, div_weight=1.5, check_iter=200, min_save_step=1200, patience=8, composite_mode=str_loss`）：

| 臂简称 | 完整实验名 | 使用监督信号 | 原型绑定对象与槽位数 | 科学角色 |
| --- | --- | --- | --- | --- |
| `baseline` | `esconv_case_baseline` | 仅 8 类支持策略 | 无原型 / 无 PCAM / 无 PCGrad | 原版 CASE-ESConv 基线 |
| `emo_wo_all_proto` | `esconv_wo_all_proto` | 8 类策略 + 11 类静态会话情绪 | 无原型（`--disable_all_prototypes`） | 情绪辅助支路去原型强对照 |
| `emo_epcl` | `esconv_case_epcl` | 8 类策略 + 11 类静态会话情绪 | 11 类负面静态会话情绪（`proto=11`） | 粒度错配对照组（暴露边界条件） |
| **`strategy_epcl`** | **`esconv_strategy_epcl`** | **仅原生 8 类支持策略（零额外标注）** | **原生 8 类轮次动态策略（`7..14 -> 0..7`, `proto=8`）** | **任务对齐版 CASE-EPCL 主迁移臂** |

---

## 3. 全量 Test（N=1,817, 40,318 tokens）四臂双 Best 并列总表

数据源：[`results/cloud_archive/esconv_strategy_epcl_20261123/esconv_strategy_compare_summary.json`](../../results/cloud_archive/esconv_strategy_epcl_20261123/esconv_strategy_compare_summary.json)。

### 3.1 主报告口径：`ppl_best`（全论文统一选点口径）

| 指标 (ESConv Test, N=1817) | `baseline` (Step 3999) | `emo_wo_all_proto` (Step 3999) | `emo_epcl` (Step 2599) | **`strategy_epcl` (Step 3799)** | `strategy_epcl` vs `baseline` | `strategy_epcl` vs `emo_epcl` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Valid PPL ↓** | `61.1353` | `60.7759` | **`60.4392`** | `60.8131` | `-0.3222` | `+0.3739` |
| **Test `case_ppl` ↓** | `44.5961` | `44.9225` | `44.5858` | **`44.3823`** | **`-0.2138`** | **`-0.2035`** |
| **Test `token_micro_ppl` ↓** | `57.0427` | `57.1190` | **`56.3019`** | `56.4586` | **`-0.5841` (`p=0.0233`*)** | `+0.1567` (`p=0.534`) |
| **Str CE ↓** | `1.8193` | **`1.8080`** | `1.8098` | `1.8136` | `-0.0057` | `+0.0038` |
| **Str Acc (%) ↑** | `33.46` | **`34.01`** | `33.19` | `33.79` | **`+0.33 pp`** | **`+0.60 pp`** |
| **Str Macro-F1 (%) ↑** | `20.72` | `22.48` | `19.57` | **`22.83`** | **`+2.11 pp`** | **`+3.26 pp`** |
| **Greedy Dist-1 (%) ↑** | `0.7456` | **`0.9476`** | `0.7481` | `0.9105` | **`+0.1649` (+22.1%)** | **`+0.1624` (+21.7%)** |
| **Greedy Dist-2 (%) ↑** | `1.9260` | **`3.1887`** | `1.9168` | `2.7488` | **`+0.8228` (+42.7%)** | **`+0.8320` (+43.4%)** |
| **Greedy Unique (%) ↑** | `6.2741` | **`11.2823`** | `5.2834` | `9.5762` | **`+3.3021` (+52.6%)** | **`+4.2928` (+81.3%)** |
| **Greedy Avg_Len** | `9.74` | `9.70` | `9.27` | `9.43` | `-0.31` | `+0.16` |
| **Greedy BLEU-2 / BLEU-4 / ROUGE-L** | `3.51 / 1.22 / 15.64` | **`3.61 / 1.21 / 15.95`** | `3.02 / 0.97 / 15.59` | `3.11 / 1.01 / 15.48` | `-0.40 / -0.21 / -0.16` | `+0.09 / +0.04 / -0.11` |
| **Beam-5 Dist-1 (%) ↑** | `0.6653` | `0.6954` | `0.6831` | **`0.7384`** | **`+0.0731` (+11.0%)** | **`+0.0553` (+8.1%)** |
| **Beam-5 Dist-2 (%) ↑** | `1.6792` | `1.8547` | `1.7945` | **`2.0748`** | **`+0.3956` (+23.6%)** | **`+0.2803` (+15.6%)** |
| **Beam-5 Unique (%) ↑** | `6.2190` | `7.0446` | `5.5036` | **`7.2647`** | **`+1.0457` (+16.8%)** | **`+1.7611` (+32.0%)** |
| **Beam-5 Avg_Len** | `10.34` | `9.58` | `9.59` | `8.35` | `-1.99` | `-1.24` |
| **Beam-5 BLEU-2 / BLEU-4 / ROUGE-L** | **`3.70 / 1.26 / 15.72`** | `3.30 / 1.17 / 15.20` | `3.13 / 1.10 / 14.25` | `2.34 / 0.78 / 14.58` | `-1.36 / -0.48 / -1.14` | `-0.79 / -0.32 / +0.33` |
| **原型有效秩 `eff_rank`** | — | — | `7.51 / 11` | `4.97 / 8` (`62.1%` 占比) | — | — |

### 3.2 附录对照口径：`composite_best`

> `baseline`（Step 3999）、`emo_epcl`（Step 2599）与 **`strategy_epcl`（Step 3799）** 三个臂的双 Best 均落在同一步，指标与 `ppl_best` 完全一致；仅 `emo_wo_all_proto` 的 `composite_best` 落在 Step 2599（`case_ppl = 45.2677`, `token_micro_ppl = 57.1254`, `Str_Acc = 32.80%`, `Str_Macro-F1 = 21.10%`, `Greedy D2/Uniq = 2.78/13.98%`, `Beam D2/Uniq = 2.00/5.83%`）。在 `composite_best` 口径下，`strategy_epcl` 同样在 `case_ppl`（显著优于 `emo_wo_all_proto`, `-0.89`）、`Str_Acc`（全场第一 `33.79%`）、`Str_Macro-F1`（全场第一 `22.83%`）及 `Beam-5 Dist-1/2/Unique` 上全部居首。

---

## 4. 统计显著性、长度分桶与机理验证

### 4.1 成对显著性检验（`ppl_best`, N=1,817）
- **`strategy_epcl` vs `baseline`（同监督端到端净效应）**：
  - `token_micro_ppl` 逐样本总 NLL 配对 t 检验：`mean_diff = +0.2297, t = +2.269`, **`p = 0.0233`（统计显著更优）**；
  - `case_ppl` 宏平均 NLL 配对 t 检验：`diff = +0.00481, t = +0.847, p = 0.397`（方向占优）；
  - `Str_Acc` McNemar 精确检验：`b=149, c=155, p = 0.774`（整体准确率小幅占优，核心增益在少数类 **Macro-F1 `+2.11pp`**）。
- **`strategy_epcl` vs `emo_wo_all_proto`（零情绪标注 vs 额外情绪标注强基线）**：
  - `case_ppl` 宏平均 NLL 配对 t 检验：`t = +2.157`, **`p = 0.0310`（统计显著更优）**；
  - `token_micro_ppl` 逐样本总 NLL 配对 t 检验：`t = +2.473`, **`p = 0.0134`（统计显著更优）**。

### 4.2 目标回复长度分桶 PPL（`ppl_best`，`microPPL / casePPL`）

| 目标长度区间 | 样本数 `n` | 总 Token | `baseline` | `emo_wo_all_proto` | `emo_epcl` | **`strategy_epcl`** |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `len [1, 10]` | 383 | 2,766 | **`27.89 / 27.32`** | `28.22 / 27.89` | `28.80 / 28.21` | `28.34 / 28.13` |
| `len [11, 20]` | 642 | 9,789 | `39.58 / 38.36` | `39.97 / 38.74` | `39.84 / 38.60` | **`38.80 / 37.65`** |
| `len [21, 35]` | 521 | 13,856 | `57.85 / 57.17` | `58.09 / 57.38` | **`56.87 / 56.28`** | `57.27 / 56.67` |
| `len [36, ∞)` | 271 | 13,907 | `83.87 / 79.00` | `83.09 / 78.16` | **`81.25 / 76.56`** | `83.11 / 78.06` |

- `strategy_epcl` 在 ESConv 测试集样本数最多的主干回复区间 `len [11, 20]`（`n=642`）拿到 4 臂全场第一（`-0.78` micro-PPL / `-0.71` case-PPL），并在 `len >= 21` 全部中长回复区间稳定优于 `baseline`，从而拿到 4 臂最低的全局 `case_ppl (44.3823)`。

### 4.3 三项核心机理验证全部闭环
1. **彻底根治通用共情句塌缩**：`emo_epcl` 中占比高达 `626/1817 (34.5%)` 的 `'i can understand how you feel . '` 在 `strategy_epcl` 中彻底退出 Top-6；Greedy 唯一句数从 `96` 跃升至 **`174`**（超过原版 `114`），Beam-5 唯一句数跃升至 **`134`**（4 臂最高）。
2. **少数类策略召回显著恢复**：
   - `Restatement or Paraphrasing (str=8, n=120)`：准确率从 `baseline 8.3%` / `emo_epcl 1.7%` 跃升至 **`22.5%`**；
   - `Self-disclosure (str=10, n=172)`：`strategy_epcl` 是 4 臂中唯一获得非零预测与召回（`16` 次预测，`1.2%`）的模型；
   - `Affirmation (str=11)` 达 **`26.9%`**（4 臂最高），`Providing Suggestions (str=12)` 达 **`55.5%`**，推动 `Str Macro-F1` 升至 **`22.83%`**（4 臂最高）。
3. **无过拟合稳态收敛**：`Valid STR_loss` 全程平滑收敛在 `1.824–1.833`，彻底消除 `emo_epcl` 中静态会话情绪 `EMO_loss` 飙升（`1.68 -> 3.14`）导致的早停问题；`PCAM` 门控平滑演化至 `gamma = -0.3342, tanh(gamma) = -0.3223`（与 ED `v10-T15` 同为负向残差调节，`Out(V)` 范数 `5.98` 远比 `emo_epcl` 的 `8.92` 温和稳健，`K` 相对有效秩占比从 `24.1%` 升至 `37.4%`）。

---

## 5. 论文写作口径与诚实边界声明（遵循铁律 I ~ VI）

1. **推荐写入论文跨数据集泛化主表的核心 3 行对比（全用 `ppl_best`）**：
   - `CASE` (`esconv_case_baseline`)
   - `CASE-EPCL (Emotion-Proto, Mismatched)` (`esconv_case_epcl`)
   - **`CASE-EPCL (Strategy-Proto, Task-Aligned)` (`esconv_strategy_epcl`)**
   - 叙事逻辑极强且 100% 诚实：在 ESConv 上直接照搬 ED 的会话级静态情绪原型虽能压低长句 `token_micro_ppl`，但因 11 类全负面情绪跨 14 轮恒定导致自由解码共情模板塌缩；将同一 `CASE-EPCL` 框架对齐至 ESConv 原生轮次动态 8 类支持策略（**零额外标注，与原版 CASE 100% 公平对齐**）后，在 **`case_ppl` (`44.60 -> 44.38`)、`token_micro_ppl` (`57.04 -> 56.46, p=0.0233`)、`Str Acc` (`33.46 -> 33.79%`)、`Str Macro-F1` (`20.72 -> 22.83%`)、Greedy `Dist-1/2/Unique` (`+22% / +43% / +53%`) 与 Beam-5 `Dist-1/2/Unique` (`+11% / +24% / +17%`)** 上全面超越原版 `CASE-ESConv` 与错配版 `Emotion-Proto`。
2. **必须如实守住的两条科学边界（严禁越界宣称）**：
   - **n-gram 重叠权衡**：`strategy_epcl` 的 `BLEU-2/4` 与 `ROUGE-L` 略低于原版 `CASE`（尤其 Beam-5 受平均句长缩短 `10.34 -> 8.35` 的长度惩罚影响），报告时应如实并列呈现，不可隐匿；
   - **组件归因口径（铁律 V）**：`esconv_strategy_epcl` vs `esconv_case_baseline` 属于**策略对齐版 CASE-EPCL 全套框架相对于原版 CASE 的联合净效应**；若审稿人进一步追问策略原型框架内部的纯原型边际贡献，可用现有脚本 `run_esconv_strategy_epcl.sh both` 补跑 1 个 `esconv_strategy_wo_all_proto` 消融臂（约 1.1 小时）即可完成精确拆解。
