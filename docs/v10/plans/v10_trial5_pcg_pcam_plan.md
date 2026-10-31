# V10 Trial 5 事前方案：后验与情境引导原型交叉注意力（PCG-PCAM，seed=13）

日期：2026-10-31。依据审计铁律 v3。定位：探索期单轮筛选实验，如实按组合改动联合净效应登记；新增云端 24G 训练 1 次（预计约 78～82 分钟）。

## 1. 立项依据与为什么可能兼顾四项指标

- 本地只读审计已证伪分类头静态融合（DPFC `p=0.76`、三路编码特征残差探测净变 `-6～-18` 题），故分类头保持原共享 `emotion_linear` 不变；同时鉴于 `Trial2 (clip=0.0)` 在 `Acc (41.83%)` 与 `Greedy D2 (18.12%)` 上相对 `T11-B (clip=6.0)` 明显退化，本轮底座直接切回全周期综合基本盘最优的 **`V9 T11-B (clip=6.0)`**。
- 实测定位现行 `PCAM` 结构性坍缩：`dec_pre_logit` 二范数（`17.51`）与单位原型（`1.00`）相差 `17.5` 倍且未接入编码端原型后验 `P_route`（熵 `0.72` vs `PCAM` 熵 `3.43/3.47`），致注意力退化为 `98.9%` 均匀平均（跨样本余弦 `0.9271`），前 500 条回复上下文实词命中率仅 `19.04%`（人类 `47.70%`）。
- `PCG-PCAM`（`--use_pcg_pcam`）对同一模块实施三项协同修复：① 将 32 个全局原型与当前对话情境槽 `[concept_enc, cs_enc]`（排除 `fine_emotion` 防辅助头恒等拷贝捷径）统一 L2 归一化并乘以 `sqrt(d_model)=17.3205` 构建 `[B,34,300]` 同尺度记忆；② 前 32 槽注入编码端可导路由对数先验 `softplus(route_beta)*log_softmax(proto_logits)`，后 2 槽配零初始化偏置 `ctx_slot_bias`，保留 `gamma=0` 冷启动；③ 同步 `train_one_batch` / `Greedy` / `Sampling` / `Beam` 四路解码（含 Beam 活跃样本切片）。
- 四项兼顾逻辑：借 `clip=6.0` 守住高分类与低情绪损失基本盘，同时消除词表映射前均匀底噪并注入对话专属概念/常识记忆以压低 `PPL`（补齐 `T11-B` 唯一短板）并推高 `Greedy Dist-1/2`，词级 `ctx_loss` 经可导路由先验协同强化 `fine_emotion` 原型归属以争取 `Acc↑`。

## 2. 配置（单模块对照基线：V9 T11-B clip=6.0）

| 实验 / 对照 | 唯一模块差异 | 其余恒定配置 |
|---|---|---|
| `v9_trial11_B_gnorm6`（已有基线） | 原版 `PCAM`（`[32,300]` 单位原型，无路由先验与情境槽） | `seed=13`、共享头、**`clip=6.0`**、`Arc ramp=2500`、`MCP=0.96`、`lambda_epcl=0.03`、`PCGrad` |
| `v10_trial5_pcg_pcam` | 启用 `--use_pcg_pcam`（`[B,34,300]` 同尺度混合记忆 + 可导路由对数先验，仅增 6 个一维标量参数） | 同上（4 轮预训练及主训练 Step 0 前向与 `T11-B` 一致） |

## 3. 事前判定、继续与暂停规则（Test N=5255；生成顺序前 N=500，max_dec_step=40）

- 对照基准：直接对标 **`V9 T11-B`** 与 **`V9 T10`**——`T11-B` 本地修正路由口径（`lambda_epcl=0.03`，`beam=prototype_routing_v1`）为 `PPL 32.57`（云端 `32.56`）、`Acc 42.21%`、`EMO_loss 2.2840`、`Greedy D1/D2 5.30% / 18.77%`（云端旧口径 `5.11% / 18.19%`）、`Beam D1/D2 4.83% / 14.38%`；`T10` 为 `PPL 32.48 / Acc 42.40% / Greedy D1/D2 5.58% / 18.14%`。
- 同表并列双 Best（`composite_best` 主判，`ppl_best` 并列禁反挑），`Greedy D1/D2` 为主、`Beam D1/D2` 并列，标明联合净效应与运行环境。
- 值得继续条件：`composite_best` 四项全面越过 `T11-B` 且 `PPL` 越过 `T10`（即 **`PPL < 32.48`**、**`Acc > 42.21%`**、**`Greedy Dist-1 > 5.30%`**、**`Greedy Dist-2 > 18.77%`**），则锁定为有效候选，仅补最少必要消融后定稿。
- 暂停条件：若未实现相对 `T11-B` 的四项同向改善（或有质量/多样性取舍），直接按均值判负封盘，不搜超参、不跑 Bootstrap 或 5255 全量生成，停止架构试错转入论文撰写。

## 4. 云端命令

```bash
bash main.sh 24G v10_trial5_pcg_pcam
```
核对日志 `max_grad_norm=6.0` 及指纹 `use_pcg_pcam=true`、`pcam_mode="posterior_context_guided_v1"`、`pcam_memory_slots=34`；回传至 `results/cloud_archive/v10_trial5_pcg_pcam/`，终审写 `docs/v10/reports/v10_trial5_audit.md`（`<=55` 行）。
