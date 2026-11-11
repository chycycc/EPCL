# V10 Trial 10 事前计划：PCAM 内部 LayerNorm 恒等重构 (`v10_trial10_pcam_inner_ln`)

> **文件性质**：V10 系列正式实验计划与实施规约。依据项目 [因果审计铁律 v3](../../../.agents/rules/case-epcl-experiment-audit-rules.md)。  
> **立项日期**：2026-11-11  
> **前序基准**：直接承接 `v10_t7_masked_pool`（24G 规范，`batch=32`, `accum=1`, `max_grad_norm=0.0`, `arc_margin_ramp_steps=2500`, `use_pcmg_head`, `use_masked_react_pool`）。

---

## 1. 核心目标与要回答的问题

在严格保持 `hop=1`、独立去重参数量 `30,336,257`（零参数净增）、完全不变动训练随机性与优化器调度梯度的前提下，解决解码端原型交叉记忆注意力模块（PCAM）在初始步未达成数学严格恒等映射的问题：
1. **Test PPL** $\downarrow$：突破 32.40（击败基线 T7 composite 的 32.46 与历史参考）；
2. **Test EMO_acc** $\uparrow$：保持或超越基线（$\ge 42.36\%$）；
3. **Test EMO_loss** $\downarrow$：保持或优于基线（$\le 2.2623$）；
4. **多样性指标**：维持 Greedy $\text{D1} \ge 5.29\%,\, \text{D2} \ge 18.31\%$。

---

## 2. 背景诊断与数学机制（方案 B）

### 2.1 原版 PCAM 的数学缺陷与隐状态畸变
原版 PCAM（V6~V10）的融合计算定义为：
$$
h_{\text{enhanced}} = \mathrm{LN}\Big(h_{\text{dec}} + \tanh(\gamma) \cdot \big(g \odot h_{\text{proto}}\big)\Big)
$$
其中 $h_{\text{dec}}$ 为 DecoderLayer 输出，$\gamma$ 为零初始化标量（$\tanh(0)=0$），$g$ 为通道门控。  
在理论设想中，$\gamma=0$ 应实现残差冷启动保护。然而，由于外层包裹了全局 $\mathrm{LN}(\cdot)$：
$$
\left. h_{\text{enhanced}} \right|_{\gamma=0} = \mathrm{LN}(h_{\text{dec}})
$$
事实上，$h_{\text{dec}}$ 在送入 PCAM 前已在第 1 层 DecoderLayer 末尾完成了一次标准 LayerNorm。再次施加全局 LayerNorm 会强制重构通道分布的均值与方差，导致初始步隐状态发生非恒等畸变，并在反向传播时引入不必要的二次归一化梯度扰动。

### 2.2 方案 B：PCAM 内部 LayerNorm 恒等重构
方案 B 将 LayerNorm 从外层残差和外移入原型微扰分支内部：
$$
h_{\text{enhanced}} = h_{\text{dec}} + \tanh(\gamma) \cdot \Big(g \odot \mathrm{LN}\big(h_{\text{proto}}\big)\Big)
$$
- **严格数学恒等性**：当 $\gamma=0$ 时，$\tanh(0)=0$，第二项精确置零：
  $$
  \left. h_{\text{enhanced}} \right|_{\gamma=0} \equiv h_{\text{dec}}
  $$
  实现初始步零漂移、零隐状态畸变的纯净直通。
- **原型特征尺度受控**：$\mathrm{LN}(h_{\text{proto}})$ 仅对跨注意力上下文输出进行归一化，使微扰增量具有稳定的单位方差，避免门控放大引发的数值溢出。

---

## 3. 本地 11 组 1,500 Updates 前置筛选实测全景

在正式上云前，项目在本地 Windows 环境严格以 `result_Base_scratch.json`（同一种子 seed=13、同一确定性数据流、`bsz=4, accum=8`）为基准，对候选方案进行了 1,500 updates（0/500/1000/1500 检查点，全量 Valid $N=5734$ TF + $N=256$ Greedy）的前置严谨筛选。

### 1,500 Updates 终点实测对比总表
数据归档：[`docs/v10/trial10_screening/`](file:///e:/github/CASE/docs/v10/trial10_screening/)

| 实验臂 | 核心改动说明 | twPPL | Δ twPPL | Acc (%) | Δ Acc (pp) | dec_emo_acc (%) | D1 / D2 (%) | 裁决结论 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Base_scratch** | **共同基准基线（hop=1）** | **51.86** | - | **40.41** | - | **39.43** | 4.75 / 12.28 | **基线** |
| `Cand_tie` | D1 词嵌入与输出权重绑定 | 70.16 | +18.30 | 38.61 | -1.80 | 19.72 (-19.71) | 0.00 / 0.00 | **严重恶化，淘汰** |
| `Cand_dec2` | D4 解码器增至 2 层 (`hop=2`) | 49.39 | -2.46 | 41.68 | +1.27 | 40.58 (+1.15) | 5.23 / 13.40 | **降级为容量附表**（改变参数量 +1.02M，留作后续 Scaling 验证） |
| **`Cand_optB`** | **方案 B (`--pcam_inner_ln`)** | **51.04** | **-0.82** | **42.15** | **+1.74** | **40.11 (+0.68)** | **4.79 / 11.65** | **全维度胜出，正式立项 T10** |
| `Cand_pcmg1` | PCMG 原型校准分支 detach | 51.04 | -0.82 | 40.91 | +0.51 | 39.20 (-0.23) | 4.55 / 11.10 | 弱于 OptB，淘汰 |
| `Cand_pcmg2` | 子句原型显著性注意力池化 | 51.86 | -0.00 | 41.51 | +1.10 | 39.52 (+0.09) | 4.31 / 9.88 | PPL 无收益且 D2 暴跌，淘汰 |
| `Cand_pcmg3` | 解码端辅助头对称原型校准 | 50.98 | -0.88 | 40.77 | +0.37 | 37.57 (-1.87) | 4.96 / 11.75 | 辅助头准确率崩塌，淘汰 |
| `Cand_optB_pcmg1`| OptB 叠加 PCMG-1 | 50.97 | -0.89 | 41.58 | +1.17 | 38.98 (-0.45) | 4.83 / 11.55 | 较 OptB 仅多降 0.08 PPL，但 Acc 回落 0.58pp，淘汰 |
| `Cand_optB_syn1` | 分类头注入 VAD 概念 (ST) | 51.73 | -0.13 | 40.76 | +0.35 | 38.44 (-0.99) | 5.18 / 14.02 | 较 OptB twPPL 反弹 +0.68，Acc 暴跌 -1.40pp，淘汰 |
| `Cand_optB_syn2` | 编解码端分类双向软蒸馏 | 51.36 | -0.50 | 41.66 | +1.26 | 39.52 (+0.09) | 4.49 / 10.32 | 较 OptB 双核心指标均退化，淘汰 |
| `Cand_optB_syn3` | 分类对数后验先验偏置注入 | 51.86 | -0.00 | 41.04 | +0.63 | 39.36 (-0.07) | 4.28 / 9.83 | 完全回吐 OptB 收益回退 Base，淘汰 |

### 筛选结论
1. **方案 B 表现出卓越的单变量正向因果效应**：在参数量不增（30,336,257）、显存仅微增 0.6 MB 的极简约束下，实现困惑度改善（twPPL -0.82）与主干情绪分类准确率大幅提振（Acc +1.74pp，dec_emo_acc +0.68pp）；
2. **否定强行跨模块耦合**：PCMG 分支修改及概念图/分类头对生成注意力的前向直通，均会干扰解码分布的自回归判别空间，引发指标回退。架构解耦与纯净恒等残差是保持共情与困惑度均衡的关键基石。

---

## 4. 实验配置与云端执行规范

### 4.1 唯一受控自变量
- 命令行开关：`--pcam_inner_ln`（`action="store_true"`，显式开启）；
- 其它所有开关严格置 `False`（包括已彻底从生产代码中精简清理的 `pcmg_detach_proto_cal` 等）；
- `hop=1`、`backbone_dropout=0.0`。

### 4.2 云端 Profile 声明（`main.sh`）
在 `main.sh` 中配置标准 24G Profile：
```bash
    v10_trial10_pcam_inner_ln|v11_optB_pcam_inner_ln)
        if [ "$ENV_MODE" != "24G" ]; then
            echo "[V10 T10] This profile requires 24G (batch32/accum1); the schedule is not mapped to 4G." >&2
            exit 1
        fi
        if [ -n "$EXTRA_FLAGS" ]; then
            echo "[V10 T10] This named profile fixes EXTRA_FLAGS; unset inherited overrides first." >&2
            exit 1
        fi
        if [ -e "$OUTPUT_DIR" ] || [ -e "$LOG_FILE" ]; then
            echo "[V10 T10] Output already exists: $EXP_NAME; refusing to overwrite." >&2
            exit 1
        fi
        EXTRA_FLAGS="--max_grad_norm 0.0 --arc_margin_ramp_steps 2500 --use_pcmg_head --save_snapshots_from 6499 --use_masked_react_pool --pcam_inner_ln"
        ;;
```

### 4.3 云端启动命令
```bash
bash main.sh 24G v10_trial10_pcam_inner_ln
```
产物自动落盘并路由至：`results/v10/`，快照保存至 `save/v10_trial10_pcam_inner_ln/snapshots/`。

### 4.4 架构指纹硬约束（铁律 IV）
模型初始化末尾单行输出必须严格匹配：
```json
[Architecture Fingerprint] {"backbone_dropout": 0.0, "dec_layers": 1, "actual_backbone_dropout_p": 0.0, "actual_dec_layers": 1, "pcam_inner_ln": true, "use_main_emo_ce_anneal": false, "main_emo_ce_anneal_start": 5000, "main_emo_ce_anneal_end": 8500, "main_emo_ce_anneal_min_weight": 0.2, "main_emo_ce_anneal_scope": "train_only", "use_pcg_pcam": false, "pcam_mode": "legacy_pcam", "pcam_memory_slots": 32, "pcam_mem_scale": 1.0, "use_emo_group_lr": false, "emo_lr_anneal_start": 5000, "emo_lr_anneal_end": 6500, "emo_lr_min_scale": 0.05, "emo_branch_params": 0, "main_branch_params": 30336257, "use_pcmg_head": true, "use_emotion_routing": false, "use_masked_react_pool": true, "pcmg_split_lambda": 0.15, "pcmg_proto_weight": 0.25, "pcmg_proto_tau": 0.2}
```

---

## 5. 验收红线与裁决标准

以官方测试集（$N=5255, \text{batch}=1$）实测为准：
- **基准对照**：`v10_t7_masked_pool`（PPL 32.46, Acc 42.36%, EMO_loss 2.2623, D1 5.39%, D2 18.51%）；
- **保持底线容差（铁律 III）**：
  - Test PPL $\le 32.56$（基线 $+0.10$）；
  - Test Acc $\ge 42.21\%$（基线 $-0.15\text{pp}$）；
  - Greedy D1 $\ge 5.29\%$（基线 $-0.10\text{pp}$）；
  - Greedy D2 $\ge 18.31\%$（基线 $-0.20\text{pp}$）；
- **通过判据**：在守住上述四项底线的前提下，实现 Test PPL 显著下降（$\le 32.40$）或 Test Acc 明显提升（$\ge 43.0\%$）。若指标全面跌破底线，即刻判定为失败封盘，严禁事后调整超参或过度调试。
