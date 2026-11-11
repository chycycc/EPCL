# V11 分类与生成协同（Classification-Generation Synergy）深度剖析与实测裁决报告

## 1. 历史实验边界与源码级断层诊断

结合 `docs/v8/`、`docs/v9/`、`docs/v10/` 历史实验记录与 [model.py](file:///src/models/CASE/model.py) 前向源码，当前 CASE-EPCL 在“情绪分类”与“共情生成”之间存在三处结构性断层：

| 断层位置 | 源码现状与问题机制 | 历史实验教训（必须规避的坑） |
| :--- | :--- | :--- |
| **断层 1：分类头看不见 VAD 情感概念特征** | 在 [model.py#L2256-L2258](file:///src/models/CASE/model.py#L2256-L2258) 中，上下文词与 ConceptNet 概念已乘上人类情感词典 **VAD（效价-唤醒度-优势度）权重** 编码为 `concept_enc`，并融合进生成端的 `emotion_enc`（[L2418](file:///src/models/CASE/model.py#L2418)）。但主分类头输入 `cls_emotion_feat`（[L2447](file:///src/models/CASE/model.py#L2447)）仅读取 `fine_emotion`（COMET 短语瓶颈），完全丢弃了 `concept_enc`。 | **V8 T1–T3 教训**：直接把 `cls_anchor` 设为 `emotion_enc` 会让分类交叉熵梯度回传进 `concept_enc` 与 `emo_gate`，干扰概念图编码器并拉高 PPL。需测试对 `concept_enc` 分支做梯度截断（Stop-Gradient）。 |
| **断层 2：解码端“后验情绪”未反哺编码端“先验分类”** | 训练期解码器输入包含真实目标回复 `dec_batch_shift`（Teacher Forcing），因此解码器隐状态池化 `pooled_dec`（[L2636](file:///src/models/CASE/model.py#L2636)）实质上是看到真实回复后的“后验情绪表征”。但目前两者仅各自拟合 One-Hot 硬标签，缺乏“后验指导先验”通道。 | **V8.2 F2 教训**：强行放大硬标签分类损失权重会迫使隐状态坍缩至单点 One-Hot，撕裂生成流形（PPL 恶化 `+1.05`）。需测试温度软化的 KL 分布协同。 |
| **断层 3：主分类头 `emotion_logits` 在生成时是“死胡同叶节点”** | 核查 [model.py#L2489-L2615](file:///src/models/CASE/model.py#L2489-L2615) 与 `decoder_greedy`（[L2990-L3050](file:///src/models/CASE/model.py#L2990-L3050)）发现：主分类头的输出 `emotion_logits` 仅用于计算 `emotion_loss` 和打印准确率，从未输入给生成解码器或 PCAM。 | **V9 T9 与 V10 T5 教训**：严禁离散 Top-1 硬路由；V10 T5（`use_pcg_pcam`）因将原型向量乘 $\sqrt{300}=17.32$ 倍且未截断路由梯度而崩溃。需测试原尺度下截断梯度的软对数先验偏置。 |

---

## 2. 三项分类-生成协同单变量方案定义（基于 `Cand_optB` 父基准，`hop=1`，零新增参数）

1. **候选 S1（`Cand_optB_syn1`：`--pcam_inner_ln + --cls_concept_st`）**：
   $$\text{cls\_emotion\_feat} = \text{fine\_emotion} + 0.25 \cdot (\text{emotion\_enc} - \text{fine\_emotion}).\text{detach}()$$
2. **候选 S2（`Cand_optB_syn2`：`--pcam_inner_ln + --enc_dec_emo_kl`）**：
   训练期（$T=2.0$）在 `emotion_loss` 上叠加 $0.10 \cdot T^2 \cdot \text{KL}(P_{\text{dec}}^{\text{detach}} \| P_{\text{enc}})$，并将 `dec_emo_loss` 设为 $0.80 \cdot \text{CE} + 0.20 \cdot T^2 \cdot \text{KL}(P_{\text{enc}}^{\text{detach}} \| P_{\text{dec}})$。
3. **候选 S3（`Cand_optB_syn3`：`--pcam_inner_ln + --pcam_cls_prior`）**：
   在 PCAM 注意力权重上叠加无梯度分类对数先验偏置：$\text{attn\_weights} + 0.20 \cdot \log \text{softmax}(\text{emotion\_logits.detach()})$。

---

## 3. 本地 1,500 Updates 实测结果与证伪归因

数据源：[result_Base_scratch.json](file:///scratch/t11_weight_tying/result_Base_scratch.json)、[result_Cand_optB.json](file:///scratch/t11_weight_tying/result_Cand_optB.json)、[result_Cand_optB_syn1.json](file:///scratch/t11_weight_tying/result_Cand_optB_syn1.json)、[result_Cand_optB_syn2.json](file:///scratch/t11_weight_tying/result_Cand_optB_syn2.json)、[result_Cand_optB_syn3.json](file:///scratch/t11_weight_tying/result_Cand_optB_syn3.json)。

### 3.1 分检查点（500 / 1000 / 1500）完整轨迹对比表

| 检查点 | 指标 | Base_scratch | **Cand_optB（直接父基准）** | Cand_optB_syn1 (`cls_concept_st`) | Cand_optB_syn2 (`enc_dec_kl`) | Cand_optB_syn3 (`pcam_cls_prior`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **500** | **twPPL / bPPL** | 72.48 / 67.34 | 71.58 / 66.47 | 71.12 / 66.01 | 70.47 / 65.48 | 70.60 / 65.60 |
| | **EMO_loss** | 2.2049 | **2.1927** | 2.2008 | 2.2008 | 2.1887 |
| | **Acc / dec_emo_acc (%)** | 36.31 / 31.17 | **37.25 / 29.82** | 36.71 / 30.64 | 36.62 / 29.56 | 37.02 / 30.01 |
| **1000** | **twPPL / bPPL** | 57.14 / 53.33 | **55.88 / 52.15** | 56.37 / 52.61 | 56.80 / 53.02 | 57.37 / 53.57 |
| | **EMO_loss** | 2.1369 | **2.0981** | 2.1272 | 2.1184 | 2.1281 |
| | **Acc / dec_emo_acc (%)** | 38.19 / 35.04 | **39.20 / 34.71** | 38.39 / 34.22 | 38.68 / 36.08 | 38.35 / 35.49 |
| **1500** | **twPPL / bPPL** | 51.86 / 48.14 | **51.04 / 47.34** | 51.73 / 47.98 (`+0.68 / +0.64`) | 51.36 / 47.58 (`+0.32 / +0.24`) | 51.86 / 48.04 (`+0.81 / +0.71`) |
| | **EMO_loss** | 2.0726 | **2.0227** | 2.0477 (`+0.0250`) | 2.0342 (`+0.0116`) | 2.0611 (`+0.0385`) |
| | **Acc (%)** | 40.41 | **42.15** | 40.76 (`-1.40`) | 41.66 (`-0.49`) | 41.04 (`-1.12`) |
| | **dec_emo_acc (%)** | 39.43 | **40.11** | 38.44 (`-1.67`) | 39.52 (`-0.59`) | 39.36 (`-0.75`) |
| | **D1 / D2 (%)** | 4.75 / 12.28 | **4.79 / 11.65** | 5.18 / 14.02 (`+0.39 / +2.37`) | 4.49 / 10.32 (`-0.30 / -1.33`) | 4.28 / 9.83 (`-0.50 / -1.82`) |

### 3.2 证伪归因与核心机制结论

1. **`Cand_optB_syn1`（`--cls_concept_st`）证伪归因**：
   - 即便截断了对 `concept_enc` 的反向梯度，在前向中混入 `0.25 * emotion_enc` 仍将 ConceptNet 一跳常识概念图中的泛化噪声带入了 32 类细粒度情绪分类器（印证了 V8 T4 的发现：`concept_enc` 适合提供生成词汇多样性——故 `D2` 升至 `14.02%`，但对 32 类精确情绪分类是噪声源），导致 `Acc` 下降 `-1.40pp`、并通过共享 `emotion_linear` 反噬了 `twPPL (+0.68)`。
2. **`Cand_optB_syn2`（`--enc_dec_emo_kl`）证伪归因**：
   - 解码端 `dec_emo_logits` 与编码端 `emotion_logits` 共享同一个线性矩阵 `emotion_linear`，在训练中前期两者预测错误率均在 `60%` 左右；双向互蒸馏造成错误软分布自激，削弱了硬标签对共享分类头的监督锐度，导致 `twPPL (+0.32)` 与 `Acc (-0.49pp)` 双退化。
3. **`Cand_optB_syn3`（`--pcam_cls_prior`）证伪归因**：
   - 将句子级分类对数后验 `0.20 * log_softmax(emotion_logits.detach())` 广播加到解码器每个 Token 的 PCAM 原型注意力上，强行压制了不同位置 Token 根据自身局部语义选择不同原型的自由度（而句子级 Top-1 分类本身有 ~58% 错判率），直接抵消了 `Cand_optB` 全部的 PPL 收益（`twPPL` 从 `51.04` 退回 `51.86`）并使 `D2` 跌至 `9.83%`。

---

## 4. 本阶段 9 组 1,500 Updates 从头训练实验终局总结表

| 实验臂代号 | 架构与开关定义 | `hop` | 参数量 | twPPL ↓ | EMO_loss ↓ | Acc (%) ↑ | dec_emo_acc (%) ↑ | D1 / D2 (%) | 最终裁决 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Base_scratch`** | V10 T7 + `masked_react_pool` | 1/1 | 30.34M | 51.86 | 2.0726 | 40.41% | 39.43% | 4.75 / 12.28 | 基准对照 |
| **`Cand_optB`** | **`+ --pcam_inner_ln`（方案 B）** | **1/1** | **30.34M** | **51.04 (`-0.82`)** | **2.0227 (`-0.050`)** | **42.15% (`+1.74`)** | **40.11% (`+0.68`)** | **4.79 / 11.65** | **唯一全面胜出（上云首选）** |
| `Cand_tie` | `+ weight_sharing=True`（D1） | 1/1 | 22.94M | 70.16 (`+18.30`) | 2.0775 (`+0.005`) | 40.95% (`+0.54`) | 19.72% (`-19.71`) | 2.92 / 8.46 | 淘汰 |
| `Cand_pcmg1` | `+ --pcmg_detach_proto_cal` | 1/1 | 30.34M | 51.04 (`-0.82`) | 2.0717 (`-0.001`) | 40.91% (`+0.51`) | 39.20% (`-0.23`) | 4.55 / 11.10 | 被 `Cand_optB` 覆盖，淘汰 |
| `Cand_pcmg2` | `+ --pcmg_proto_salience_pool` | 1/1 | 30.34M | 51.86 (`-0.00`) | 2.0564 (`-0.016`) | 41.51% (`+1.10`) | 39.52% (`+0.09`) | 4.31 / 9.88 | PPL 零收益，淘汰 |
| `Cand_pcmg3` | `+ --pcmg_dec_proto_cal` | 1/1 | 30.34M | 50.98 (`-0.88`) | 2.0638 (`-0.009`) | 40.77% (`+0.37`) | 37.57% (`-1.87`) | 4.96 / 11.75 | 解码分类退化，淘汰 |
| `Cand_optB_pcmg1` | `optB + pcmg_detach_proto_cal` | 1/1 | 30.34M | 50.97 (`-0.89`) | 2.0349 (`-0.038`) | 41.58% (`+1.17`) | 38.98% (`-0.45`) | 4.83 / 11.55 | 劣于纯 `Cand_optB`，淘汰 |
| `Cand_optB_syn1` | `optB + cls_concept_st` | 1/1 | 30.34M | 51.73 (`-0.13`) | 2.0477 (`-0.025`) | 40.76% (`+0.35`) | 38.44% (`-0.99`) | 5.18 / 14.02 | 劣于纯 `Cand_optB`，淘汰 |
| `Cand_optB_syn2` | `optB + enc_dec_emo_kl` | 1/1 | 30.34M | 51.36 (`-0.50`) | 2.0342 (`-0.038`) | 41.66% (`+1.26`) | 39.52% (`+0.09`) | 4.49 / 10.32 | 劣于纯 `Cand_optB`，淘汰 |
| `Cand_optB_syn3` | `optB + pcam_cls_prior` | 1/1 | 30.34M | 51.86 (`-0.00`) | 2.0611 (`-0.012`) | 41.04% (`+0.63`) | 39.36% (`-0.07`) | 4.28 / 9.83 | 劣于纯 `Cand_optB`，淘汰 |
| `Cand_dec2` | `+ --dec_layers 2`（D4） | 1/2 | 31.36M | 49.39 (`-2.46`) | 2.0365 (`-0.036`) | 41.68% (`+1.27`) | 40.58% (`+1.15`) | 5.23 / 13.40 | 仅作 Scaling 附表备选 |
