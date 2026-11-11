# V11 阶段优化方向筛选与上云准备报告（D1 / D2 / D4）

## 1. 真实瓶颈诊断与前期排查结论

通过对 [model.py](file:///src/models/CASE/model.py)（V10 T7/T9 配置）前向数据流、损失函数与梯度投影机制的源码核查，排除并更正了以下假设，定位出尚未触及的主干容量与正则化瓶颈：

| 排查维度 | 核查事实与结论 | 关键证据与代码位置 |
| :--- | :--- | :--- |
| **知识选择 Train/Eval 一致性** | **完全一致（排除泄漏）**。训练、验证与贪婪解码下游均统一使用先验注意力（`prior_concept_enc` / `prior_cs_enc`）；依赖目标回复的后验分布仅用于计算 `bow_loss` 与 `kl_loss`。 | [model.py#L2276-L2317](file:///src/models/CASE/model.py#L2276-L2317) |
| **多任务梯度冲突（方案 C：分层 PCGrad）** | **非真实瓶颈（排除）**。云端 T7 全程 10,000 步中负内积步数占比为 `37.16%`，但冲突步平均余弦绝对值仅 $\|\cos\| = 0.0220$（夹角约 $91.26^\circ$，近似正交），PCGrad 投影修正量极微。 | `v10_trial7_pcmg_head.log` L29988；[audit_cases.md](file:///docs/audit_cases.md) 案例 7.2 |
| **生成主干容量与正则化缺失（真实瓶颈）** | 1. **主干极浅**：编码器与解码器均为 `num_layers=1, heads=2, hidden=300, filter=300`（[config.py#L75-L78](file:///src/utils/config.py#L75-L78)）。<br>2. **主干零 Dropout**：全局 `config.dropout=0.1` 未传入 [`make_encoder`](file:///src/models/CASE/model.py#L1853-L1863) 与 `self.decoder`，内部四类 Dropout 全部保持默认值 `0.0`，导致云端训练在 7,499 步见顶后过拟合。<br>3. **参数分布失衡**：词嵌入与输出层各占约 `7.39M` 参数（词表 `24,647 × 300`），合计占总参数量约 `48.7%`，而解码器单层仅约 `1.02M` 参数。 | [model.py#L1375-L1398](file:///src/models/CASE/model.py#L1375-L1398)<br>[model.py#L1853-L1863](file:///src/models/CASE/model.py#L1853-L1863) |

> [!WARNING]
> **新增本地训练工程陷阱（已归档至 [audit_cases.md](file:///docs/audit_cases.md) 案例 7.1）**
> [model.py#L1546](file:///src/models/CASE/model.py#L1546) 初始化底层 Adam 时将 `lr` 设为 `0`，而训练步进在 [model.py#L2720](file:///src/models/CASE/model.py#L2720) 直接调用底层 `opt.step()` 绕过了 `NoamOpt.step()`。任何独立训练脚本必须复刻 [main.py#L108-L117](file:///main.py#L108-L117) 的主阶段学习率显式注入逻辑（`config.lr = 0.000576`），否则参数将静默保持零更新。

---

## 2. 方向 D1（输出层与词嵌入权重绑定 `weight_sharing=True`）：已证伪并关停

- **实验设置**：唯一自变量 `config.weight_sharing`（`Base=False` vs `Cand_tie=True`），同随机种子、同确定性批次序列从头训练 `1,500` updates（`bsz=4, accum=8`，有效批 32）。
- **1,500 步实测结果**（[result_Base_scratch.json](file:///scratch/t11_weight_tying/result_Base_scratch.json) vs [result_Cand_tie.json](file:///scratch/t11_weight_tying/result_Cand_tie.json)）：
  - `twPPL`：`51.8594 → 70.1571`（**劣化 +18.2976**）
  - `bPPL`：`48.1409 → 65.0083`（**劣化 +16.8674**）
  - `Acc`：`40.41% → 40.95%`（`+0.54pp`）
  - `dec_emo_acc`：`39.43% → 19.72%`（**严重退化 −19.71pp**）
  - `D1 / D2`：`4.75% / 12.28% → 2.92% / 8.46%`（**退化 −1.83pp / −3.82pp**）
- **机制归因与裁决**：
  1. **初始 Logit 尺度失配**：绑定后输出层直接使用未除以 $\sqrt{d}$ 的 GloVe 词向量，Step-0 `twPPL` 高达 $9.45 \times 10^{11}$，前期梯度被迫用于压缩词向量模长，与编码端输入（乘 $\sqrt{300}$）形成拉扯。
  2. **解码端隐状态双重目标冲突**：解码器输出 `pre_logit` 同时被绑定后的词嵌入空间与共享分类头 `emotion_linear`（`dec_emo_loss`）争夺，导致解码端情绪分类准确率腰斩。
  - **裁决**：未过前置门槛，直接判负关停。

---

## 3. 方向 D4（解码器扩容至 2 层 `--dec_layers 2`）：通过本地门槛，推荐优先上云

### 3.1 代码修改与零漂移校验
- **新增开关**：在 [config.py#L309-L313](file:///src/utils/config.py#L309-L313) 与 [model.py#L1375-L1397](file:///src/models/CASE/model.py#L1375-L1397) 新增 `--backbone_dropout`（默认 `0.0`）与 `--dec_layers`（默认 `None`，沿用 `config.hop=1`），并在 [common.py#L303](file:///src/models/common.py#L303) 修正 `DecoderLayer` 返回值为 8 元组以支持多层 `nn.Sequential` 透传 `cs_enc_outputs` 与 `concept_enc_outputs`。
- **兼容性与评测模式核查**：
  - 全仓检索确认 `src/`、`tests/` 与 `benchmarks/`（独立 `sys.path`）无任何解包冲突；
  - 确认 [main.py#L142](file:///main.py#L142)、[eval_pipeline.py#L355](file:///src/scripts/eval_pipeline.py#L355) 等全部评测与贪婪解码入口均处于 `model.eval()` 模式。
- **Step-0 零漂移断言**：默认参数（`backbone_dropout=0.0, dec_layers=None`）与 `backbone_dropout=0.1` 在 Step-0 全量 Valid（`N=5734`）下相对 `result_Base_scratch.json` 的 `twPPL / EMO_loss / Acc` 差值均为 `0.000000e+00`。

### 3.2 本地 1,500 Updates 从头训练对比表（`Base_scratch` vs `Cand_dec2`）

数据源：[result_Base_scratch.json](file:///scratch/t11_weight_tying/result_Base_scratch.json) 与 [result_Cand_dec2.json](file:///scratch/t11_weight_tying/result_Cand_dec2.json)（脚本：[run_t11_arm.py](file:///scratch/t11_weight_tying/run_t11_arm.py)）。

| 检查点 (Updates) | 实验臂 | twPPL | bPPL | EMO_loss | Acc (%) | dec_emo_acc (%) | D1 (%) | D2 (%) | PCG 冲突率 (%) | PCG \|cos\| | 峰值显存 (MB) | 累积耗时 (min) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Base** (`dec_layers=1`) | 24790.12 | 24809.53 | 3.8210 | 2.51 | 3.45 | 12.32 | 20.95 | 0.00 | 0.0000 | 1492.2 | 2.01 |
| | **Cand_dec2** (`dec_layers=2`) | 24694.45 | 24714.00 | 3.8210 | 2.51 | 4.01 | 16.31 | 31.20 | 0.00 | 0.0000 | 1510.9 | 2.94 |
| | *差值* | *-95.67* | *-95.53* | *0.0000* | *0.00* | *+0.56* | *+3.99* | *+10.25* | *0.00* | *0.0000* | *+18.7* | *+0.93* |
| **500** | **Base** | 72.48 | 67.34 | 2.2049 | 36.31 | 31.17 | 2.36 | 4.09 | 39.40 | 0.0534 | 2333.3 | 15.22 |
| | **Cand_dec2** | 67.28 | 62.44 | 2.2099 | 37.44 | 31.23 | 2.05 | 3.71 | 40.12 | 0.0550 | 2359.7 | 18.86 |
| | *差值* | *-5.20* | *-4.90* | *+0.0050* | *+1.13* | *+0.07* | *-0.31* | *-0.38* | *+0.72* | *+0.0016* | *+26.4* | *+3.64* |
| **1000** | **Base** | 57.14 | 53.33 | 2.1369 | 38.19 | 35.04 | 2.52 | 6.03 | 39.06 | 0.0544 | 2333.3 | 28.36 |
| | **Cand_dec2** | 53.10 | 49.46 | 2.1245 | 38.39 | 36.57 | 3.19 | 7.94 | 39.73 | 0.0544 | 2359.7 | 34.68 |
| | *差值* | *-4.04* | *-3.87* | *-0.0124* | *+0.20* | *+1.53* | *+0.68* | *+1.91* | *+0.67* | *0.0000* | *+26.4* | *+6.32* |
| **1500** | **Base** | 51.86 | 48.14 | 2.0726 | 40.41 | 39.43 | 4.75 | 12.28 | 39.72 | 0.0538 | 2333.3 | 41.65 |
| | **Cand_dec2** | 49.39 | 45.75 | 2.0365 | 41.68 | 40.58 | 5.23 | 13.40 | 40.14 | 0.0537 | 2359.7 | 50.43 |
| | **1500 净差值** | **-2.46** | **-2.39** | **-0.0361** | **+1.27** | **+1.15** | **+0.49** | **+1.12** | **+0.42** | **-0.0001** | **+26.4** | **+8.78** |

### 3.3 客观风险提示
1. **前期收敛加速 vs 渐近收益**：`twPPL` 差值在 `500 / 1000 / 1500` 步分别为 `-5.20 / -4.04 / -2.46`（相对改善从 `7.17%` 收窄至 `4.75%`）。10,000 步完整收敛后的终态净收益必须通过云端全量训练确认。
2. **单种子与初始化偏移**：增加第 2 层解码器改变了初始化时的 RNG 消耗序列，编码器与分类头的初始权重与 `Base` 存在随机偏移；按铁律 III，待云端单变量全量训练达标后再决定是否补充显著性或多种子检验。
3. **参数量说明**：总参数量从 `30,336,257` 增至 `31,359,857`（净增 `+1,023,600`，约 `+3.37%`）。

---

## 4. 云端待跑 Profile 清单与执行指令

在 [main.sh](file:///main.sh) 中已配置完成以下两个严格单变量 Profile（均通过本地 20 步冒烟与指纹核验，结果目录路由至 `results/v11/`）：

1. **优先实验（D4：解码器 2 层）**：
   - 启动命令：`bash main.sh v11_d4_dec2`
   - 唯一自变量：`--dec_layers 2`（其余参数与 `v10_t7_masked_pool` 逐字一致）
   - 架构指纹：`{"backbone_dropout": 0.0, "dec_layers": 2, "actual_backbone_dropout_p": 0.0, "actual_dec_layers": 2, "main_branch_params": 31359857, "use_pcmg_head": true, "use_masked_react_pool": true, ...}`
2. **并行/次选实验（D2：主干开启 Dropout 0.1）**：
   - 启动命令：`bash main.sh v11_d2_backbone_dropout`
   - 唯一自变量：`--backbone_dropout 0.1`（作用于 5 个 Transformer Encoder 与 1 个 Decoder，针对 7,499 步后的过拟合）
   - 架构指纹：`{"backbone_dropout": 0.1, "dec_layers": 1, "actual_backbone_dropout_p": 0.1, "actual_dec_layers": 1, "main_branch_params": 30336257, "use_pcmg_head": true, "use_masked_react_pool": true, ...}`

**云端裁决标准（铁律 II & III）**：提取官方 `eval_metrics.json`，并列报告 `composite_best` 与 `ppl_best`；在 ED Test 全量（`batch=1, shuffle=False, N=500 Greedy`）上对比 T9 基线（`PPL = 32.4577, Acc = 44.7383%, D1 = 5.2686%, D2 = 18.2275%`）。
