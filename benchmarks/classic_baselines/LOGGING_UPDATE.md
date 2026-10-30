# 云端日志显示更新

2026-10-30。更新四模型的控制台显示：tqdm 进度条、速度/预计剩余时间、具名训练指标、Valid 指标及 TensorBoard。CASE 显示 BOW/KL/MIM/CTX/PPL/EMO_loss/EMO_acc；其他三模型显示 Loss/PPL/EMO_loss/EMO_acc。准确率在控制台显示为百分比，JSON/TensorBoard 保留原始 0–1 数值。

训练目标、batch、种子、更新次数、数据顺序、早停与选点规则不变。完整控制台仍由 train.sh 写入 `runs/formal_<model>.log`；JSON 原字段保留并增加 `native_metrics`，TensorBoard 写入运行目录的 `tensorboard/`。这是原生指标和进度展示的恢复，不是切换回作者训练脚本。

将 `baseline_logging_update_20261030.tar.gz` 上传至云端 `classic_baselines_bundle/`，执行：

```bash
cd /root/autodl-tmp/baseline_reproduction/classic_baselines_bundle
tar -xzf baseline_logging_update_20261030.tar.gz
```

包只更新 `baseline_training_v1/` 下的训练显示代码、查看工具、说明和 Git 版本记录，不包含配置、数据或权重。正在运行的 CASE 继续使用已加载代码；新显示在下次启动/续训时生效，不要为日志重启当前训练。

在另一个终端实时查看当前 CASE 的具名日志：

```bash
python baseline_training_v1/show_log.py \
  --run-dir runs/formal_case_original_v2 --follow
```

Ctrl+C 只退出这个查看命令。去掉 `--follow` 可一次输出已有记录，也可重定向为文本文件。它只读取 JSONL，不调用模型。旧 Valid 文件只有 PPL，就只显示已有 PPL；没有记录过的历史验证指标不能还原。

后续模型使用原命令即可，例如：

```bash
bash baseline_training_v1/train.sh multi_trs
```

测试：具名指标与百分比映射，以及原有选点、Valid 聚合、批次恢复和训练续训共 5 项通过。没有重新训练四模型。
