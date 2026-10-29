# V10 Trial 4 事前方案：后期主分类 CE 余弦衰减（seed=13）

日期：2026-10-29。依据审计铁律 v3。已实现，CPU单步sanity与Shell语法检查通过；云端基线待补评登记，尚未训练。

## 1. 要回答的问题

noclip底座的主分类Valid CE在约5000步后升高，而生成PPL仍在下降。本轮检验：后半程降低主分类监督，能否缓解CE恶化，同时保住PPL与Acc？

## 2. 配置

| 实验 / 对照 | 唯一差异 | 其余 |
|---|---|---|
| Trial2 noclip（已有权重） | 5000步后主CE权重保持1.0 | seed=13、split_data_seed=13 |
| `v10_trial4_main_ce_anneal` | 5000→8500步主CE权重从1.0余弦降至0.2，此后保持0.2 | 同上 |

- 调度按从0计数的主训练iter执行：5000步前额外系数w=1；5000～8500步`w=0.2+0.4*(1+cos(pi*(iter-5000)/3500))`；之后w=0.2，不含预训练。
- 仅训练时对主CE乘w；保留2500～5000步、峰值1.25的钟形增强。EPCL系数0.03、辅助CE系数0.1不变；Valid/Test仍报告未加权CE。
- 固定共享头、noclip、Arc cos/margin0.30/ramp2500；MCP、PCAM、PCGrad及其他损失和学习率沿用Trial2。云端24G、batch32/accum1/fp32、预训练4轮；保持max_step10000、check_iter500、patience6、min_save_step3000、Valid shuffle=False及`composite_score=PPL+5*EMO_loss`；保留全部实际验证点。
- 实施仅涉及config.py、model.py和main.sh：新增默认关闭的`--use_main_emo_ce_anneal`及start/end/min_weight参数，不复用会同时衰减EPCL的`use_cosine_anneal`；一行架构指纹、一次必要sanity后提交推送。

## 3. 判定规则（事前登记，单种子筛选）

- 训练前在同一云端环境补评Trial2双Best，composite为`CASE_7499_37.2898_composite_best`；结果独立保存至`results/v10/trial4_baseline_noclip/`并登记基线数值，保留原归档，无需重训。
- 双方统一修正评测：lambda_epcl=0.03、beam=prototype_routing_v1；Test5255/batch1/shuffle=False，顺序生成500/batch1/max_dec_step40，核采样作辅助；沿用skip_test、skip_manifold和串行双Best流程。
- 以Valid选出的composite、云端JSON两位报告值裁决：**PPL不高于基线、Acc不低于基线，且至少一项严格改善**，记为单种子候选；否则判负、保留原noclip配置。不拿本地参考32.30/41.83%或Trial3双种子均值作硬门槛。
- 同表并列双Best的PPL、Acc、EMO_loss及Greedy/Beam Dist-2、Unique；ppl不用于反挑。对比共同Valid步数的原始CE，判断后期恶化是否缓解；通过不等于跨种子有效或回复更具体。
- 硬指标判负即封盘，不追加bootstrap或全量生成；本轮只跑seed13一组，不同时开展B/C及其他超参搜索。

## 4. 云端命令

```bash
python src/scripts/eval_pipeline.py --model_path results/cloud_archive/v10_trial2_noclip --eval_target both --metrics_path results/v10/trial4_baseline_noclip/eval_metrics.json --results_path results/v10/trial4_baseline_noclip/eval.txt --save_path results/v10/trial4_baseline_noclip --lambda_epcl 0.03 --max_grad_norm 0 --use_arc_margin --arc_margin_ramp_steps 2500 --skip_manifold --gen_samples 500 --batch_size 32 --gpu 0
bash main.sh 24G v10_trial4_main_ce_anneal
```
以上先补评（双Best置于model_path目录），登记基线后再启动训练；核对架构指纹中的开关及5000/8500/0.2。回传至`results/cloud_archive/v10_trial4_main_ce_anneal/`，结果只写`docs/v10/reports/v10_trial4_audit.md`（≤55行）并同步总账。
