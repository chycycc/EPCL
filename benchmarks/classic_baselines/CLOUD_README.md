# 云端准备包：只做环境核验与推理 smoke

2026-10-26 后续：用户已回传四模型 GPU 推理通过，实际共同环境为 Python 3.10.8 / PyTorch 2.1.2+cu121 / RTX 4090D。下文保留原准备包的推理步骤；新增独立训练烟测及增量上传说明见 [TRAINING_SMOKE_README.md](TRAINING_SMOKE_README.md)，正式训练尚未启动。

本包包含四模型独立源码、统一数据包和检查工具。**没有训练子命令，不含模型权重，不代表正式训练已验收。**本地验证是 Python 3.8 / torch 1.12 的 CPU 随机权重 smoke；云端目标根据用户回传与修订授权更新为 Python 3.10 / torch 2.1.2 / CUDA 12.1。GPU 名称兼容 `4090D` 与 `4090 D`；修订后仍须重跑预检与 GPU 推理 smoke。

将 `.tar.gz` 上传至云端的新目录并解压，进入 `classic_baselines_bundle/`。以下命令应在实际 CASE-EPCL 使用的同一 Python 环境执行：

```bash
python tools/verify_cloud_bundle.py --bundle-root .
python tools/cloud_preflight.py --output case_epcl_cloud_environment.json
python -m unittest discover -s tools/tests -v
python tools/run_smoke_queue.py --workspace-root workspace --data-dir data --output-dir runs/cloud_inference_smoke --device cuda --reference-environment case_epcl_cloud_environment.json
```

四条均不训练。最后一条串行检查四模型，各用 2 条真实样本、随机权重、最多生成 3 token、beam=2；保存临时随机检查点并验证恢复，实际优化器更新数必须为 0。输出目录必须是新路径。NLTK 数据已有自定义位置时追加 `--nltk-data /actual/nltk_data`。

完成后保留 `case_epcl_cloud_environment.json`、`runs/cloud_inference_smoke/queue.json` 和四份 `.log`、模型子目录中的 `metadata.json`、`records.jsonl`。随机权重指标不能作为基线性能。

正式模型仍保持原生结构与损失；统一协议使用当前 CASE 缓存的分词、24,647 词表和缓存知识特征。KEMP 的知识输入由同一缓存适配，未冒充作者原始预处理数据。打包清单中的 `external_glove` 仅为后续正式初始化的外部文件指纹，约 1GB 的 GloVe 没有重复打包。

云端推理 smoke 通过后，还需验证预训练词向量初始化、训练反向和更新路径，并冻结每模型训练预算、验证与早停配置，随后才提供正式训练启动命令。CASE-EPCL 既有训练代码和 V10 方案不由本包修改。
