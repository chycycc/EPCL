# IAMM 双版本、ESCM 与 SEEK

四个独立配置：`iamm`、`iamm_no_situation`、`escm`、`seek`。与 CASE-EPCL 生产代码隔离，复用现有基线的云端环境、GloVe、确定性 Valid、训练状态恢复和日志工具。云端只训练，回传后在本地评测。

## 源码与输入

| 配置 | 官方仓库 / 版本 | 输入和训练监督 |
|---|---|---|
| IAMM | [zhouzhouyang520/IAMM](https://github.com/zhouzhouyang520/IAMM)，8b23e15 | 原生 Situation 文本与其 COMET |
| IAMM 去 Situation | 同一版本、同一初始化与超参 | Train/Valid/Test 均将 Situation 替换为15个固定UNK；五类Situation-COMET均替换为固定UNK序列，原架构保留；这是场景输入消融 |
| ESCM | [官方仓库](https://github.com/zhouzhouyang520/EmpatheticDialogueGeneration_ESCM)，65e729d | 作者依存句法树、关系词表与原生模型；不输入 Situation |
| SEEK | [wlr737/EMNLP2022-SEEK](https://github.com/wlr737/EMNLP2022-SEEK)，cd4e5d2 | 作者带41类逐话语情绪/意图标签的缓存，保留原生FACE；额外监督在结果中标注 |

SEEK统一Acc读取32类对话情绪头；原生逐话语、目标话语分类准确率另外保存在native_returns.jsonl，不将41类准确率填入32类对比列。各模型采用作者缓存、分词与词表，在本地评测时按canonical Test的对话和目标顺序对接；标点分词边界差异按拼接文本对应。PPL保留原生目标编码并记录实际token数和词表大小；聚合公式一致不代表编码条件相同。不伪造句法或41类标签。

兼容修改仅在工作副本：np.float改为np.float64，GPU硬编码改为config.device，GloVe改逐行读取。IAMM原单条Greedy函数与当前forward不匹配，采用训练结构对应的decoder_greedy_batch并固定batch=1；短序列统一PAD至top-k所需15位置，COMET至少2位置。双版本使用同一collate。Greedy步数按原循环的max_dec_step+1换算；Beam-5从同一原生Greedy分布取得上下文。

## 云端训练

共同环境：Python3.10.8、PyTorch2.1.2+cu121、CUDA12.1、RTX4090D 24GB。seed=13、batch16/accum1/FP32、Noam warmup8000；无BOW预训练；每2000步Valid，14000步起选优，patience3，最多30000步。Valid以exp(mean(batch token CE))选优，IAMM/ESCM原生返回的逐样本CE按实际token数还原batch CE。训练主循环沿用七基线的统一配方，各模型保留原生训练目标。

把 `extended_baselines_20261118.tar.gz` 上传到已有bundle根目录：

```bash
cd /root/autodl-tmp/baseline_reproduction/classic_baselines_bundle
tar -xzf extended_baselines_20261118.tar.gz
OMP_NUM_THREADS=8 bash extended_training_v1/train.sh iamm
OMP_NUM_THREADS=8 bash extended_training_v1/train.sh iamm_no_situation
OMP_NUM_THREADS=8 bash extended_training_v1/train.sh escm
OMP_NUM_THREADS=8 bash extended_training_v1/train.sh seek
```

四条命令依次运行。GloVe默认 `/root/autodl-tmp/CASE/vectors/glove.6B.300d.txt`，已有共享data目录复用；包内带三项目源码与作者预处理ED缓存，不带GloVe和模型权重。中断后在原命令后加 `--resume`。

可选的短运行命令为 `bash extended_training_v1/train.sh iamm --smoke-updates 1`，必须用 `BASELINE_RUN_DIR` 指定独立目录；不占用正式目录。它只运行指定次数更新、状态保存恢复与一条样本的分类/NLL/Greedy/Beam。

每组保存 `last.pt`、`ppl_best.pt`、run.json、train.jsonl、valid.jsonl、training_result.json、tensorboard/、native_metrics.log、native_tensorboard/、native_returns.jsonl；完整stdout/stderr位于相邻formal_<model>.log。

## 回传和本地评测

云端训练结束后打包：

```bash
tar -czf /root/autodl-tmp/extended_baseline_returns.tar.gz \
  runs/formal_iamm runs/formal_iamm.log \
  runs/formal_iamm_no_situation runs/formal_iamm_no_situation.log \
  runs/formal_escm runs/formal_escm.log \
  runs/formal_seek runs/formal_seek.log
```

本地解压至 `results/benchmarks/extended_baselines/cloud_return/`，保留包内runs层级。在CASE根目录运行，例如IAMM：

```bash
E:/Dev/condaData/envs_dirs/cem_env/python.exe benchmarks/extended_baselines/run.py iamm \
  --workspace-root benchmarks/extended_baselines/workspace \
  --shared-data results/benchmarks/classic_baselines/phase2_20261026/data \
  --run-dir results/benchmarks/extended_baselines/cloud_return/runs/formal_iamm \
  --embedding-file E:/github/CASE/vectors/glove.6B.300d.txt \
  --evaluate-only --gen-samples 5255 --max-dec-step 50
```

其他组替换模型名和formal目录。Test N5255、batch1、shuffle=False；全量Greedy/Beam-5与此前本地50步表对齐。eval_metrics.json同时保存case_ppl=exp(mean_i CE_i)、sample_macro_ppl=mean_i exp(CE_i)、token_micro_ppl，主对比取case_ppl。日常前500条／40步可用默认参数。每条目标NLL不含PAD、含EOS，不含辅助损失。

源代码准备入口prepare.py、云端包生成入口pack.py；缓存来自IAMM作者ED.zip、ESCM作者data.zip及SEEK仓库dataset_preproc.p。IAMM双版本用于检验场景输入效应，效应方向与幅度由结果决定。

本地已完成四配置各一次实际反向更新和训练状态恢复；IAMM原版完成分类/NLL与原生Greedy，去Situation、ESCM、SEEK另完成共同搜索对原生Greedy的一致性及Beam短运行。使用本地Python3.8/PyTorch1.12 CPU与现有GloVe；未启动云端正式训练。运行输出保存在scratch/extended_*.txt，不随云端包上传。
