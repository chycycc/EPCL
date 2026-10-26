# 四模型独立训练烟测（2026-10-26）

本入口仅做有界工程验收，不提供 epoch、正式训练步数或自动续跑选项。用户已回传四模型云端 GPU 推理全部通过，共同环境为 Python 3.10.8 / PyTorch 2.1.2+cu121 / RTX 4090D。训练 GPU 验收须执行下列新命令。

## 增量上传与执行

只上传 `training_smoke_tools_20261026_v1.tar.gz` 到已有目录 `/root/autodl-tmp/baseline_reproduction/classic_baselines_bundle/`。包内是新工具目录，保留既有 `tools/`、`workspace/`、`data/`、环境快照与推理产物。包含旁支的 CASE 准确率标量兼容、4090 D 名称及 PyTorch 2.1.2 环境修补。不含数据、GloVe、权重，无需重新上传大准备包。

在已通过推理验收的 CASE-EPCL Python 环境运行。`case_epcl_cloud_environment.json` 使用该环境已有的有效快照（torch 2.1.2），不要使用旧目标快照。

```bash
cd /root/autodl-tmp/baseline_reproduction/classic_baselines_bundle
tar --keep-old-files -xzf training_smoke_tools_20261026_v1.tar.gz
python training_smoke_tools_20261026_v1/verify_cloud_bundle.py --bundle-root training_smoke_tools_20261026_v1
python -m unittest discover -s training_smoke_tools_20261026_v1/tests -v
python training_smoke_tools_20261026_v1/run_training_smoke_queue.py \
  --workspace-root workspace \
  --data-dir data \
  --output-dir runs/cloud_training_smoke_v1 \
  --embedding-file /root/autodl-tmp/CASE/vectors/glove.6B.300d.txt \
  --device cuda --batch-size 2 \
  --reference-environment case_epcl_cloud_environment.json
```

如 NLTK 数据不在默认搜索目录，最后一条追加 `--nltk-data /root/nltk_data`。每次使用新输出目录；失败保留日志，定位后再换新目录运行。

GloVe 固定 SHA256：`91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`。初始化前校验全文件，在原生加载和 Xavier 初始化后核验全部命中词表行的 float32 权重逐位相等。未命中行保持原生随机初始化策略，不修改云端原文件。

## 执行边界与断言

- 顺序 CASE → Multi-TRS → CEM → KEMP；FP32、seed=13、batch=2、accum=1。保留原生结构、目标函数、Adam/Noam，记录学习率、warmup 和实际 zero_grad 行为。
- 每模型先运行 2 次主训练更新，保存完整状态，再继续 1 次作为参考；随后**新进程**恢复并重放该下一步。CASE 额外先执行 1 次原生 BOW 预训练更新，仅覆盖此分支，不代表完整 4 轮预训练。
- 参考续步和恢复重放都会调用 optimizer.step：CASE 合计 **5 次**，其余各 **4 次**，队列合计 **17 次真实更新**。连续轨迹是 CASE 4 次、其余 3 次，不能把重放算作连续进度。默认使用首 6 条 Train 样本；batch=1..4 可选，无增加步数的参数。
- 捕获真实 backward 标量；损失、所有现存梯度、权重、Adam 状态须有限。主训练 embedding、分类头、decoder 梯度非零；embedding 和分类头须实际更新。CASE BOW 分支不要求分类头/decoder 梯度。
- 保存权重、Adam、Noam 步数/速率/配方、Python/NumPy/torch CPU/CUDA RNG、word_freq、criterion weight/true_dist、模式、计数；校验数据、源码、GloVe、环境和工具哈希。恢复后的完整状态须**逐位相等**。
- 下一步重放报告完整状态及目标值误差。浮点 tensor 固定 `atol=1e-6, rtol=1e-5`；离散状态、RNG、Noam、词频等精确一致。容差仅用于基线续步烟测，不改变 V10 冻结容差；失败停止，不自动放宽。
- 恢复后只读评估 1 条 Valid 并核验状态不变；**Test N=0、生成 N=0**。不选点、不早停、不调用正式训练 main。
- `training_smoke_checkpoint.pt` 为专用完整训练状态格式，标为 `training_smoke_only`，不能作为正式基线结果来源；保留 `expected_continuation.pt` 复核续步。

## 回传

通过标志为最后的 `all_passed: true`、`actual_optimizer_updates: 17`。保留整个 `runs/cloud_training_smoke_v1/`。先回传小文件包：

```bash
tar -czf cloud_training_smoke_v1_evidence.tar.gz \
  --exclude='*.pt' runs/cloud_training_smoke_v1
```

其中包括 queue.json、8 份日志、每模型 initial/resume 的 environment.json 和 metadata.json。权重留在云端，按记录的 SHA256 追溯。通过后再冻结正式配方和训练入口，本命令不会启动正式训练。
