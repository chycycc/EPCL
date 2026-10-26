# CASE-EPCL V10 实验总账（`v10_experiment_log.md`）

> **系列定位**：V10 系列专注于**现有组件的改造与训练配合优化**，不再继续堆叠新模块。
> **对照基线**：明确锚定 **V9 T11-B**（`v9_trial11_B_gnorm6`），不因开启 V10 系列而重置评价标准或隐藏历史结果。
> **方法论规约**：严格执行 [`.agents/rules/case-epcl-experiment-audit-rules.md`](../../.agents/rules/case-epcl-experiment-audit-rules.md) v2 的五条审计铁律（I 因果对齐、II 验证基准确定性、III 指标口径显式化〔含多指标并列报告〕、IV 开关生效性验证、V 单变量控制）。
> **工程隔离原则**：目录整理与模型改动分开处理；单轮实验不顺带删除旧组件或重构训练逻辑，统一通过 [`main.sh`](../../main.sh) 启动。

> **云端回传状态（2026-10-26）**：V10 Trial1 正式 Test `PPL=32.61 / Acc=41.75% / EMO_loss=2.4438`，未通过 `32.56 / 42.21%` 双门槛，保留 T11-B。运行时预检、9项源码清单与2份权重核验通过（loader.py仅LF/CRLF差异）。本地两份检查点Test、两组配对统计、全量5255生成及20点轨迹审计全部完成，见[报告06](reports/06_v10_trial1_cloud_results_audit.md)。

> **最新修正状态（2026-10-24）**：[03 修正与补充验收](reports/03_v10_trial1_corrections_and_acceptance.md)已完成本地闭环：61项回归、10组配置、off/on各100次实际更新及真实环境/SHA256打包通过；实际云端 `zero_grad` 行为由 `main.sh` 在训练前探测并校验。生产算法和冻结判据未改，未启动正式云端训练。

> **方案 v1.2 评测与报告层修订（2026-10-26，云端开跑前登记）**：① 新增 V9 T10、V9 T11（seed13）为必须报告的历史参照；② 新增双向判定规则（未同时追平 T10 不称历史最佳；点估计超过 T10 也须两项配对区间在改善方向排除 0，才可称所评检查点的 PPL/Acc 优于 T10；`V10 − T10` 只能报告联合净效应）；③ 生成多样性主口径改为 Test 全量 $N=5255$，由本地同环境统一补跑，云端前 500 条保留为历史连续性列。训练代码、`main.sh`、双门槛 `32.56 / 42.21%` 与选点规则均不变。配对统计不估计种子方差，不自动支持无范围限定的“历史最佳”。详见[方案 §4.2、§4.4](plans/v10_trial1_sep_dec_head_plan.md)与[04 同日复核](reports/04_v10_trial1_v1_2_protocol_review.md)。

---

## 0. 冻结对照基线（继承自 V9 收官主线）

| 实验标识 | 归档路径 / 主检查点 | Test PPL ↓<br>($N=5255$, `batch=1`, 样本 macro) | Test EMO_acc ↑<br>($N=5255$, `batch=1`, 样本 micro) | Test EMO_loss ↓<br>($N=5255$, 样本平均) | Greedy Dist-2 / Unique<br>(前 $N=500$, `batch=1`) | Beam Dist-2 / Unique<br>(前 $N=500$, `batch=1`) | 状态与说明 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **V9 T11-B**<br>(`v9_trial11_B_gnorm6`) | [`results/cloud_archive/v9_trial11_B_gnorm6/`](../../results/cloud_archive/v9_trial11_B_gnorm6/)<br>`CASE_7499_37.5058_composite_best` | **32.56**<br>(同环境本地补评 `32.564646`) | **42.21%**<br>(2218/5255，未舍入 `42.2074215%`) | **2.2840** | 18.19% / 68.00% | 14.61% / 42.60% | **当前冻结主线基线**<br>正式报告双门槛：`PPL ≤ 32.56` 且 `Acc ≥ 42.21%`（配对检验用同环境本地未舍入值） |

*历史脉络参见 [V9 实验总账 (`docs/v9/v9_experiment_log.md`)](../v9/v9_experiment_log.md) 与 [V9 Trial 12 收尾审计 (`08_prefix_norm_results_audit.md`)](../v9/reports/trial12/08_prefix_norm_results_audit.md)。*

### 0.1 历史参照（v1.2 新增，必须报告，不作为替换门槛）

口径同上表（Test $N=5255$、`batch=1`；生成为前 $N=500$ 历史连续性口径）；均为单种子 `seed=13`，`composite_best` 均在 step 7499 选出。

| 实验标识 | 梯度裁剪（实际行为） | Arc ramp | Test PPL ↓ | Test EMO_acc ↑ | Test EMO_loss ↓ | Greedy Dist-2 / Unique | Beam Dist-2 / Unique | 定位 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **V9 T10**<br>(`v9_trial10`) | 未执行（显示 2.0，调用未实装） | 无 | 32.48 | 42.40% | 2.3190 | 18.14% / 62.20% | 13.63% / 41.80% | PPL/Acc 点估计更优的历史参照；本地有权重 `CASE_7499_37.3144_composite_best` |
| **V9 T11**<br>(`v9_trial11_seed13`) | 2.0 | 2500 | 32.62 | 41.83% | 2.3396 | 18.94% / 62.00% | 13.16% / 39.00% | 裁剪分解链中间点；本地无权重 |
| **V9 T11-B** | 6.0 | 2500 | 32.56 | 42.21% | 2.2840 | 18.19% / 68.00% | 14.61% / 42.60% | 直接对照（见上表） |

- T10→T11 同时改变裁剪与 ramp，只能算联合净效应；T11→T11-B 的已核实训练配置仅改裁剪阈值 2.0→6.0，是配置层面的单变量对照；不声称历史源码与环境已逐项同一。
- **同环境补评与配对检验已完成**（2026-10-26，[报告 05](reports/05_historical_reference_local_evaluation.md)）：T10 本地 PPL/Acc/EMO_loss 为 `32.474551969 / 42.397716460% / 2.318990569`，T11-B 为 `32.564646486 / 42.207421503% / 2.283988079`。T10 − T11-B 的 95% 配对区间依次为 PPL `[-0.242593, 0.057341]`、Acc `[-0.856327, 1.236917] pp`、EMO_loss `[-0.004141, 0.074270]`；精确双侧 McNemar `p=0.748835481`。三项均未排除 0，不宣称等价或跨种子优劣；种子方差未估计。T10 本地两位 PPL 为 **32.47**，没有复现云端 **32.48**，差异来源未定位，历史正式值保持不变。
- **生成全量 $N=5255$ 主口径**：T10 与 T11-B 已完成，见下表；V10结果见本表新增行及报告06，T11 无权重，标注“未测”。500 与 5255 两种口径不得相互比较。
- **同日源码核实**：`max_grad_norm<=0` 会跳过当前裁剪，但 V10 命名配置仍锁定 6.0。T10 日志已有 `10-14 10:25` 运行记录，缺少部署源码指纹仍不能证明 F1 当时生效。归档副本按实验/检查点分目录，并校验 SHA256 与实际生成计数；04号复核时未运行新的推理或训练，后续推理结果见05号报告。

**本地同环境全量生成**（Test N=5255，batch=1，max_dec_step=40；单位 %；核采样仅作辅助记录）：

| 检查点 | Greedy Dist-2 | Greedy Unique | Beam Dist-2 | Beam Unique |
|---|---:|---:|---:|---:|
| V9 T10 composite_best | 4.35 | 36.59 | 2.90 | 18.10 |
| V9 T11（seed13） | 未测 | 未测 | 未测 | 未测 |
| V9 T11-B composite_best | 4.66 | 38.61 | 2.98 | 18.06 |
| V10 Trial1 composite_best | 4.85 | 37.05 | 3.16 | 17.22 |

两组实际生成数均为 5255，逐样本 Test 预测分别为 2228/5255、2218/5255，且均与各自云端预测逐条一致。源码、数据、原始归档和复制权重的完整性核验通过，见 [05 原始证据](reports/evidence/history_reference_20261026/verification.json)。

---

## 1. V10 实验登记总表

| 实验编号与命名 | 唯一自变量（相对基线） | 对照基线 | 预注册方案 | 云端打包目录 → 本地回传归档目录 | 当前状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V10 Trial 1**<br>(`v10_trial1_sep_dec_head`) | 主情绪分类与解码端辅助分类头解除参数共享（`--use_separate_dec_emo_head`：`emotion_linear` 与 `dec_emotion_linear` 独立参数） | **V9 T11-B** | [v10_trial1_sep_dec_head_plan.md](plans/v10_trial1_sep_dec_head_plan.md) | 云端：`~/autodl-tmp/download/v10_trial1_sep_dec_head/`<br>本地：`results/cloud_archive/v10_trial1_sep_dec_head/` | **云端训练已回传，未通过候选门槛**<br>本地补评/配对/生成/轨迹审计完成，见[06](reports/06_v10_trial1_cloud_results_audit.md) |

---

## 2. 实验分节记录

### 2.1 V10 Trial 1：主分类头与解码端辅助分类头拆分（`v10_trial1_sep_dec_head`）

- **日期**：2026-10-24（事前立项）
- **基线**：**V9 T11-B**（`v9_trial11_B_gnorm6`，Test PPL **32.56** / Acc **42.21%** / EMO_loss **2.2840**）。
- **唯一自变量**：新增 `--use_separate_dec_emo_head`，将解码端 teacher forcing 隐状态均值 `pooled_dec` 的辅助分类器从共享 `self.emotion_linear` 拆分为同源初始化的独立线性层 `self.dec_emotion_linear`（净增 $300\times 32 + 32 = 9,632$ 参数，总可学习参数 `30,336,257` → `30,345,889`）。
- **不变项**：数据集、种子（`seed=13`）、4 轮预训练、优化器与学习率退火、`max_grad_norm=6.0`、EPCL + MCP + Arc-EPCL（`ramp=2500`）+ PCAM + PCGrad、原版粗/细粒度 MIM、`alpha_mim=0.1`、`use_emo_loss_ramp=1`、Valid 选点公式与双检查点规则全部保持与 **V9 T11-B** 一致；V9 Trial 12 前缀开关全部默认关闭；不顺带删除旧组件或重构训练逻辑。
- **当前进度（2026-10-24 本地工程验收通过）**：
  - 已按冻结方案 **v1.1** [`docs/v10/plans/v10_trial1_sep_dec_head_plan.md`](plans/v10_trial1_sep_dec_head_plan.md) 完成生产代码实装与本地三阶段验收，详见 [01 工程实装与本地三阶段验收报告](reports/01_v10_trial1_implementation_and_local_verification.md) 及证据 [`results/v10/trial1_validation/verification.json`](../../results/v10/trial1_validation/verification.json)。
  - **阶段一（初始化与 4 轮预训练）**：可学习参数 `30,336,257`（`off`） vs `30,345,889`（`on`，`+9,632`）；`make_model` 与 4 轮 `pretrain` 结束后的公共模型权重、公共优化器状态及四路随机状态均**逐位相等（最大绝对差 `0.0`）**。
  - **阶段二（`batch=4, accum=1` 主训练首步等价性）**：前向各项损失与 161 个非分类头公共参数梯度**逐位相等（最大绝对差 `0.0`）**；分类头梯度加和最大绝对误差 `weight` 为 $2.98\times 10^{-8}$、`bias` 为 $1.30\times 10^{-8}$；裁剪前全局梯度范数平方恒等式 $N_{\text{split}}^2 - N_{\text{shared}}^2 = -2\langle g_{\text{enc}}, g_{\text{dec}} \rangle$ 相对误差为 **$8.19\times 10^{-11}$**；首次更新后两分类头权重即发生分离。
  - **阶段三（本地 4G `batch=4, accum=8` 100 次更新工程烟测）**：全程损失与梯度有限；`on` 组两分类头行向量余弦均值从 `1.0` 分化至 `0.963267`；`eval_pipeline.py` 与 `eval_prototype_ncm.py` 权威恢复断言全部通过（Test $N=0$，生成 $N=0$）。
  - **审查修正后补充**：完整回归现为61项；阶段三在新目录重跑，各800 microbatch/100次实际更新均有限、两个入口全部模型state tensor逐位恢复、每组9项源码及2项权重SHA逐项校验，四份检查点文件SHA与首次烟测完全一致。方案已收窄后续默认清零的表述；本地torch1.12.1的CPU/CUDA均留下零张量，24G正式入口遇此运行时会先留证再拒绝启动。[补充证据](reports/evidence/trial1_correction/verification.json)
  - **方案 v1.2 修订（2026-10-26）**：结果判读新增 V9 T10/T11 历史参照与双向判定规则，生成多样性主口径改为 Test 全量 $N=5255$（本地同环境补跑，归档前须把权重复制到独立目录，避免覆盖 `cloud_archive` 中的 `eval_metrics.json`）。训练侧无任何改动，本地三阶段验收结论继续有效。
  - **后续状态（2026-10-26）**：用户已通过入口执行云端训练并回传；历史参照补评/统计/生成完成，见[报告05](reports/05_historical_reference_local_evaluation.md)。V10主、备检查点本地Test及两组配对统计完成；全量5255生成与[报告06](reports/06_v10_trial1_cloud_results_audit.md)归档完成。不新增训练组、不切换底座。


### 2.2 V10 Trial1 正式结果与本地复核（2026-10-26）

讨论入口：[07 实验结论与下一步决策记录](reports/07_v10_trial1_findings_and_next_steps.md)。该记录建议下一轮保留共享头、单独检验`alpha_mim: 0.1→0.05`；这是待审定建议，尚未编号、修改代码或启动训练。

**结论：未通过候选主线门槛，保留 V9 T11-B。** 正式主检查点 `CASE_7499_37.6658_composite_best`：Test PPL **32.61**、EMO_acc **41.75%**、EMO_loss **2.4438**；前500生成 Greedy Dist-2/Unique、Beam Dist-2/Unique为 **19.81 / 62.80 / 13.89 / 40.40%**，全量5255见§0.1，两个口径分列。

- 主检查点同环境本地值：PPL **32.611986957**、Acc **2194/5255 = 41.750713606%**、EMO_loss **2.443770895**。V10−T11-B的配对95%区间分别为 PPL `[-0.115503, 0.208978]`、Acc `[-1.579448, 0.666032] pp`、EMO_loss `[0.116129, 0.203182]`，McNemar p=0.436068740。
- V10−T10配对95%区间分别为 PPL `[-0.020688, 0.300066]`、Acc `[-1.731684, 0.418649] pp`、EMO_loss `[0.082369, 0.167719]`，McNemar p=0.245651079；该比较包含裁剪/ramp/拆头多项变化，只能报告联合净效应。
- 两组PPL/Acc区间均包含0，EMO_loss区间均大于0。统计只针对这些检查点和测试样本，不估计种子方差，不因未显著而宣称等价；正式门槛仍按冻结舍入值判定。
- 备检查点 `CASE_8499_37.5940_ppl_best` 仅作补充：本地PPL **32.577461839**、Acc **2148/5255 = 40.875356803%**、EMO_loss **2.639457611**，未按Test改选主检查点，也未另做全量生成。
- 实际Valid各20点，共同20点，不补点。V10主检查点辅助CE=2.062184、teacher-forcing辅助Acc=44.2449%，双头行余弦均值=0.709238；不当作自由生成情绪表现或梯度冲突证明。
- 云端源码9项与冻结e97a0c0 Git blob一致；与本地相比8项逐字节一致、loader.py规范化LF后相同，两份权重SHA相符。CPU/CUDA清梯度探针通过；原始归档、生产源码、数据、权重副本前后校验通过。本地没有启动训练。

完整质量/生成并列表、全部配对区间宽度、20点Valid与诊断、源码/环境/权重证据见[报告06](reports/06_v10_trial1_cloud_results_audit.md)及[验核清单](reports/evidence/trial1_cloud_return_20261026/verification.json)。本轮归档完成，不预定Trial2。
