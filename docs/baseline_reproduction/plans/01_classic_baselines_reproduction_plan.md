# 四模型云端复现方案 v1.1

日期：2026-10-26。v1.1 补充已完成的统一数据与原生推理适配；模型范围与云端环境由用户确认。有效训练配方与云端 smoke 尚待验证，当前不是正式训练开跑冻结版。依据项目实验审计规则 v2；不修改规则文件。v1 准备审计作为历史记录保留。

## 1. 比较问题与范围

| 模型 | 比较问题 | 官方来源 | 当前实现来源 |
|---|---|---|---|
| 原版 CASE | CASE-EPCL 相对其直接基础模型的系统收益 | [作者代码](https://github.com/jfzhouyoo/CASE)、[ACL 2023 论文](https://aclanthology.org/2023.acl-long.457/) | 独立下载的作者源码，不能用当前模型关闭 EPCL 冒充原版 |
| 原版 CEM | 相对常识增强共情生成的表现 | [作者代码](https://github.com/Sahandfer/CEM) | 独立下载的作者源码，不用已优化的 `CEM-master/src/models/CEM/model.py` |
| Multi-TRS | 相对普通生成 + 情绪分类多任务学习的表现 | [CEM 官方基线实现](https://github.com/Sahandfer/CEM) | 其中 `Transformer(..., is_multitask=True)`；标注 CEM 作者复现版，不能称独立原作者实现 |
| KEMP | 相对知识图谱与情绪概念方法的表现 | [作者代码](https://github.com/qtli/KEMP) | 独立下载的作者源码 |

顺序：CASE → Multi-TRS → CEM → KEMP。各模型保留自己的结构、训练目标和合理原生训练配方。跨模型结果是系统比较，不能直接归因于 EPCL、PCGrad 或拆头；组件因果证据继续由独立消融提供。T10、T11-B 作为内部历史参考，不能取代原版 CASE。

## 2. 云端目标环境

用户指定：**Ubuntu 22.04、Python 3.10、PyTorch 2.1.0、CUDA 12.1、单张 RTX 4090D 24GB、18 vCPU AMD EPYC 9754 128-Core Processor**。

- 同一实例镜像/实际环境中运行；分别保存 Python 完整版本、torch 构建后缀、torch CUDA、cuDNN、相关依赖包、GPU 与 CPU 分配信息。
- 通过 `cloud_preflight.py` 在 CASE-EPCL 环境生成参考快照，并对比每个基线环境。CPU affinity 可能反映宿主机，额外保存 cgroup quota 并核对实例配置。
- 本地 Python 3.8 / torch 1.12 已用于数据、静态检查和随机权重 CPU 推理 smoke，不算云端 PyTorch 2.1 验收。GPU 显存峰值、耗时与训练状态均待云端实测。
- 不为迁就旧 requirements 全量覆盖既有云端环境。现代依赖兼容改动在独立副本中记录；若确实无法共用环境，先报告实际差异，不继续声称同环境。
- 第一轮统一 seed=13、FP32。训练 batch、学习率、调度和训练预算保留各模型合适配方，逐项记录并在开跑前冻结；同一环境不意味着所有模型强制相同超参。

## 3. 防耦合边界

所有新工具位于 `benchmarks/classic_baselines/`，文档位于本目录；根 `main.py/main.sh`、`src/`、`tests/` 与 sibling `E:/github/CEM-master` 保持原样。

1. 官方源码固定 commit 和逐文件 SHA256，放在套件内忽略的 `vendor/`。不向生产 `src/models` 塞入基线。
2. 每模型复制到独立 `workspace/<run>/<model>/`，保存原始/修补哈希与逐文件 diff；不覆盖已有副本。
3. 训练与推理分别以该工程自己的 cwd 和子进程运行，避免两个同名 `src`/`code` 模块在一个 Python 进程混用。未来适配器通过 JSONL、JSON、权重文件沟通。
4. 数据副本、日志、权重、缓存、生成文件按 model/run 分目录。现有 CASE/CEM 数据只读；不得把基线缓存写回根 `data/ED`。
5. 工具不调用 CASE-EPCL 的模型构造或自动磁盘清理，不改 V10 的开关、配方与双门槛，不清理任何历史权重。

## 4. 数据统一与 PPL 前置条件

以当前 CASE 的有序 context/target/emotion 为样本身份参考；已导出 `canonical/{train,valid,test}.jsonl`。ID 含 split、原始 index、内容 SHA256 前缀；完整内容 SHA256 单独保留。计数实测为 Train 40,250、Valid 5,734、Test 5,255。

已核实：CASE 与本地 CEM 的 12 个原始 `.npy` 文件字节相同；但缓存 Test 有 178 条 context、88 条 target 分词不同，词表 24,647 vs 23,712。旧缓存不能直接混用。详见准备审计报告。

统一协议与适配状态：

- 使用统一的样本顺序与目标文本；context、target、label 之外的 situation、角色、知识特征及知识生成来源单独审查。不能仅换 JSONL 表头就宣称全输入等价。
- 已生成新的只读共享包：使用 CASE 现有 context、target、emotion、situation、角色、COMET 与 ConceptNet/VAD 缓存和 24,647 词表；各模型只读取所需特征。CEM 的命名常识关系转换为原生五路列表。既有 CASE/CEM 缓存不改写。
- KEMP 输入来自这份共享概念/VAD 缓存，完整核验各 turn/token/概念特征维度；明确不是下载的作者原始 KEMP 缓存。`USR/SYS` 映射为 `[USR]/[SYS]`，沿用相同 ID、不增加词表项，结构特殊 ID 与共享词表对齐。
- 正式 PPL 排名还必须核验词表概率支持、OOV/UNK 映射、目标 token、EOS、PAD 与 pointer 分支。相同目标分词本身仍不保证相同概率事件。
- 已核验全量 Train/Valid/Test 的 context、target、concept 都在共享词表内，OOV 均为 0；四个原生 worker 的 CPU smoke 核对输出维度、目标 ID、EOS 与目标编码哈希。KEMP 保留 pointer 分布机制，当前共享数据没有扩展 OOV 输出项。云端需复核这些断言，正式表不混入作者原生词表下的 PPL。
- 不从 Test 结果选择词表、训练超参或检查点。共享词表策略如会改变原模型容量/概率支持，必须在训练前明确写出并冻结，标作统一协议下的复现而非逐项原样复刻。

## 5. 训练、选点与验收顺序

| 阶段 | 可执行内容 | 通过证据 |
|---|---|---|
| A 无 GPU 审计 | 固定官方源码，比较本地修改，导出样本指纹，整理原生配方和兼容风险 | source/data JSON、源码锁、审计报告 |
| B 无 GPU 适配 | 独立副本修补、缓存适配、模型各自的 NLL/分类/生成记录适配器、文本规范化 | 精确 diff、前后来源哈希、接口回归 |
| C 云端 smoke | 在指定云端环境验证前向/反向、有限数值、有效参数、保存恢复、两种解码、小样本导出 | 环境包、实际命令、计数、state_dict 恢复与样本记录 |
| D 正式复现 | 按冻结配方逐模型运行，独立日志与产物 | 训练日志、Valid 实际轨迹、权重/源码/数据哈希 |
| E 统一评测 | 同一 Test 顺序与指标定义，生成 500/全量两列 | 逐样本 NLL/预测、原始生成、指标与配对数据 |

基线主检查点按各自确定性 Valid PPL 选取，不以 Test 筛选；准确聚合口径、Valid batch、验证频率、开始保存时点、patience、最大预算逐模型冻结。原生脚本普遍是每 2,000 步验证、约 14,000 步才开始保存，不能套 CASE-EPCL 的 10,000 步预算而导致无有效候选；也不能让脚本的 1,000,000 上限在资源不足时无记录运行。

CASE-EPCL 既有 composite/ppl 双检查点角色保持原规则。跨模型表注明选点差异，并列呈现其现有 ppl_best 补充行以便审阅，不偷偷改为 Test 最优。原生训练方法的复现不等于单变量实验。

## 6. 统一评测合同

- Test N=5,255，batch=1、shuffle=False，校验 sample_id 完整与顺序；smoke 子集必须有单独标签与实际 N。
- NLL 只累计目标生成分布的未平滑 token CE，排除 PAD、包含 EOS；导出每样本 `nll_sum` 和实际 `token_count`。辅助、MIM、BOW、加权多样性损失均不混入。
- 同报 `mean(exp(nll_i/tokens_i))` 的样本 macro PPL 与 `exp(sum(nll_i)/sum(tokens_i))` 的 token micro PPL，先满足第 4 节的概率支持条件才排名。
- 分类导出 32 类名称、未加权 CE、正确样本数；Acc 为样本 micro，CE 为样本平均；无分类头填 null/“—”，不能填伪造的零。
- Greedy 与 Beam(k=5)，共同搜索累计 log 概率、长度惩罚 0，最多生成 40 token（含 EOS）。文本使用 NFKC + strip + TreebankWordTokenizer（保留大小写），同报 Dist-1/2、Unique、平均长度与空回复数。前 500/全量分列。CASE-EPCL 历史生成结果不得直接拼入新表；其比较输出需按相同搜索/文本规范另行导出或核验等价性，既有 V10 正式报告协议保持原样。
- 逐样本记录格式与元数据见套件 README。四模型原生推理适配器已实现，CPU 随机权重每组 2 条通过；不代替云端、完整 Test 或训练反向验收。
- 论文原表数值、作者检查点补评、我们同环境训练值分别标注；不直接拿历史论文 Dist-2 对比当前 500 条结果。

## 7. 当前状态和后续模型

四模型范围、环境要求与目录隔离已确定。官方源码、统一数据、原生推理适配和小样本 CPU smoke 已完成；正式训练配方和指定云端验收尚未完成，`cloud_ready=false`。没有启动模型训练。

准备包可先上传并在实际 CASE-EPCL 云端环境运行环境核验和推理 smoke；随后处理预训练词向量初始化、训练反向/更新验证和正式配方冻结。完成四模型后，根据论文定位补选 1–2 个近期同任务模型；CTSM 等仅为待审候选，当前不扩充训练组。
