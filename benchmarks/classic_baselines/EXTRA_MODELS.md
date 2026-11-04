# MOEL / MIME / EMPDG 云端训练

复用四模型已经使用的 Python 3.10 / PyTorch 2.1.2+cu121 / RTX 4090D 环境、数据、词表和 GloVe。
实现来源为同一 CEM 官方仓库固定版本 f45d00a3；三个独立源码目录和运行目录不接入 CASE-EPCL 生产代码。
EMPDG 使用 CEM 仓库提供的实现与训练目标，不另加其外部仓库的对抗训练流程。

上传 baseline_extra_models_20261104.tar.gz 至现有 classic_baselines_bundle 后：

```bash
cd /root/autodl-tmp/baseline_reproduction/classic_baselines_bundle
tar -xzf baseline_extra_models_20261104.tar.gz
OMP_NUM_THREADS=8 bash baseline_training_v1/train.sh moel
OMP_NUM_THREADS=8 bash baseline_training_v1/train.sh mime
OMP_NUM_THREADS=8 bash baseline_training_v1/train.sh empdg
```

三条训练命令顺序运行，每组结束自动评测。batch=16 / accum=1 / FP32 / seed=13；无 BOW 预训练；
Noam warmup=8000；每 2000 步完整 Valid；14000 步开始选优；patience=3；主训练最多30000步。
保留 MOEL/MIME 原生 scheduled oracle 训练；MOEL 验证和生成使用预测专家路由，不读取真实标签选专家。
MIME 情绪分类评测读取原生 s_weight(q_h) @ emoji_embedding.weight.T；生成使用原生 built_in 分类器。

评测沿用之前自动评测：Test N=5255、batch=1；生成前500条、Greedy与Beam-5、最大40步。
这与补充报告的全量5255条／最大50步生成结果分开标注。

每模型输出 runs/formal_<model>/：

- `native_metrics.log`：原生返回的 Loss / PPL / BCE / Accuracy 表格，准确率保留0–1原值。
- `native_tensorboard/`：作者式 loss/ppl/bce/accuracy 分组及学习率曲线。
- `train.jsonl`、`valid.jsonl`、`tensorboard/`：套件结构化日志与具名曲线；train.jsonl同时保留native_returns。
- 相邻 `runs/formal_<model>.log`：完整stdout/stderr，包括模型自己的print、进度条、训练与评测输出。
- `last.pt`、`ppl_best.pt`、`eval_metrics.json`、`test_records.jsonl`、`generation_500.jsonl`。

原作者主循环不单独再跑一次；保存的是实际本次训练的原生指标、作者式曲线和全部控制台输出。
不把套件日志称为原作者main.py独立运行的日志。

本地使用小词表CPU输入完成三个模型各一次反向更新、训练状态保存恢复、分类/NLL读取、Greedy/Beam及双日志写入。
未启动云端正式训练；本地检查不代表云端全量训练已完成。

续训／重评（替换模型名即可）：

```bash
OMP_NUM_THREADS=8 bash baseline_training_v1/train.sh moel --resume
OMP_NUM_THREADS=8 bash baseline_training_v1/train.sh moel --evaluate-only
```

指定新运行目录：`BASELINE_RUN_DIR="$PWD/runs/formal_moel_v2"` 放在 bash 命令前。
