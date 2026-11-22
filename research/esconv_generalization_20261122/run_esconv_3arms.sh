#!/bin/bash
# ==============================================================================
# ESConv 跨数据集泛化实验 3 臂全流程云端启动脚本（AutoDL 24GB RTX 4090 D）
#
# 3 臂严格控制对照（同 Session 级防泄漏 8:1:1 划分、同种子 13、同训练配方）：
#   1) esconv_case_baseline  : 原版 CASE-ESConv（无原型/PCMG/PCAM/PCGrad，仅修 ESconv 大小写 bug 与确定性评估）
#   2) esconv_wo_all_proto   : 同配方 + 11 类情绪辅助头/多粒度聚合/PCGrad/Valid-Neg，仅关闭全部 5 条原型路径
#   3) esconv_case_epcl      : 完整 CASE-EPCL 迁移版（11 类求助者情绪原型 + 保留 8 类策略主任务）
#
# 用法：
#   bash research/esconv_generalization_20261122/run_esconv_3arms.sh [all|prepare|train|eval]
# ==============================================================================
set -e
set -o pipefail

STAGE=${1:-"all"}
PYTHON_BIN=${PYTHON_BIN:-"python"}
GPU_ID=${CUDA_VISIBLE_DEVICES:-"0"}

# 中国大陆 HuggingFace 镜像，加速 COMET-ATOMIC-2020 (mismayil/comet-bart-ai2) 下载
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs results/esconv data/ESConv

if [ -d "${HOME}/autodl-tmp" ]; then
    DOWNLOAD_ROOT="${HOME}/autodl-tmp/download/esconv_3arms"
else
    DOWNLOAD_ROOT="results/download/esconv_3arms"
fi
mkdir -p "${DOWNLOAD_ROOT}"

# ==============================================================================
# 阶段 1：数据准备（幂等：若 data/ESConv/dataset_preproc.p 已存在则跳过）
# ==============================================================================
prepare_data() {
    if [ -s "data/ESConv/dataset_preproc.p" ]; then
        echo "[Prepare] 检测到 data/ESConv/dataset_preproc.p 已存在，跳过重建。"
        return 0
    fi
    if [ ! -s "data/ESConv/ESConv.json" ]; then
        echo "[Prepare] 正在拉取官方原始 ESConv.json..."
        curl -L --fail --max-time 120 -o data/ESConv/ESConv.json \
            https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json
    fi
    if [ ! -s "data/conceptnet-assertions-5.7.0.csv.gz" ]; then
        echo "[Prepare] 正在下载 ConceptNet 5.7 断言压缩包 (~474MB)..."
        curl -L --fail -C - -o data/conceptnet-assertions-5.7.0.csv.gz \
            https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
    fi
    echo "[Prepare] 启动 Session 级防泄漏划分 + 批量 COMET + ConceptNet 构图..."
    ${PYTHON_BIN} research/esconv_generalization_20261122/build_esconv_dataset.py \
        --data_dir data \
        --dataset ESConv \
        --seed 13 \
        --comet_model mismayil/comet-bart-ai2 \
        --comet_batch_size 128 \
        --assertions_gz data/conceptnet-assertions-5.7.0.csv.gz \
        --device "cuda:${GPU_ID}" 2>&1 | tee logs/esconv_prepare.log
    cp -f data/ESConv/split_audit.json data/ESConv/dataset_audit.json "${DOWNLOAD_ROOT}/"
}

# ==============================================================================
# 阶段 2：3 臂严格对齐训练
# ESConv 训练集=14,101 turns, batch_size=32 => 440.7 steps/epoch
# 按 ED (10,000 steps / 1,258 steps/epoch = 7.95 epochs) 等比例扩展至 4,400 steps (~10.0 epochs)
# ==============================================================================
COMMON_ARGS=(
    --dataset ESConv
    --gpu "${GPU_ID}"
    --seed 13
    --batch_size 32
    --accum_steps 1
    --precision fp32
    --lr 0.000576
    --warmup 800
    --lr_schedule cosine
    --lr_decay_start_step 1200
    --lr_decay_steps 2800
    --check_iter 200
    --max_step 4400
    --pretrain
    --pretrain_epoch 4
    --fine_weight 0.2
    --coarse_weight 1.0
    --div_weight 1.5
    --max_grad_norm 0.0
    --use_composite_score
    --composite_mode str_loss
    --composite_emo_loss_weight 5.0
    --min_save_step 1200
    --patience 8
    --skip_test
)

run_arm() {
    local ARM_NAME=$1
    shift
    local OUT_DIR="save/${ARM_NAME}/"
    local LOG_FILE="logs/${ARM_NAME}.log"
    if compgen -G "${OUT_DIR}/CASE_*_composite_best" > /dev/null && compgen -G "${OUT_DIR}/CASE_*_ppl_best" > /dev/null; then
        echo "[Train] 检测到 ${ARM_NAME} 双 Best 检查点已存在，跳过重复训练。"
        return 0
    fi
    echo "======================================================================"
    echo "[Train] 启动训练臂: ${ARM_NAME}"
    echo "======================================================================"
    ${PYTHON_BIN} main.py \
        "${COMMON_ARGS[@]}" \
        --exp_name "${ARM_NAME}" \
        --save_path "${OUT_DIR}" \
        --model_file_path "${OUT_DIR}" \
        "$@" 2>&1 | tee "${LOG_FILE}"
}

train_all_arms() {
    # Arm 1: 原版 CASE-ESConv 基线（关闭情绪/原型/PCMG/PCAM/PCGrad，仅保留策略主任务与相同基础调度）
    run_arm "esconv_case_baseline"

    # Arm 2: 无原型强基线（开启 11 类情绪辅助头/多粒度聚合/PCGrad/Valid-Neg/emo_branch_dropout，仅禁用全部 5 条原型路径）
    run_arm "esconv_wo_all_proto" \
        --esconv_enable_emo_epcl \
        --esconv_emo_loss_weight 1.0 \
        --use_pcmg_head \
        --use_masked_react_pool \
        --use_fine_mim_valid_neg \
        --emo_branch_dropout 0.1 \
        --use_pcgrad \
        --disable_all_prototypes \
        --use_emo_loss_ramp \
        --emo_loss_ramp_start 1000 \
        --emo_loss_ramp_steps 1000 \
        --emo_loss_ramp_max 1.25 \
        --emo_loss_ramp_shape bell

    # Arm 3: 完整 CASE-EPCL 迁移版（与 ED v10-T15 零调参同构：EPCL+MCP+MoP-DR+PCMG+PCAM+PCGrad+Valid-Neg）
    run_arm "esconv_case_epcl" \
        --esconv_enable_emo_epcl \
        --esconv_emo_loss_weight 1.0 \
        --use_mcp \
        --mcp_momentum 0.96 \
        --lambda_epcl 0.03 \
        --epcl_warmup 1000 \
        --alpha_uni 0.8 \
        --use_arc_margin \
        --arc_margin 0.30 \
        --arc_mode cos \
        --arc_margin_ramp_steps 1000 \
        --epcl_anchor fine_emotion \
        --cls_anchor default \
        --use_pcmg_head \
        --use_masked_react_pool \
        --use_fine_mim_valid_neg \
        --emo_branch_dropout 0.1 \
        --use_pcam \
        --pcam_inner_ln \
        --pcam_heads 2 \
        --pcam_dropout 0.1 \
        --pcam_gate_bias -1.0 \
        --pcam_tau_attn 1.5 \
        --use_pcgrad \
        --use_emo_loss_ramp \
        --emo_loss_ramp_start 1000 \
        --emo_loss_ramp_steps 1000 \
        --emo_loss_ramp_max 1.25 \
        --emo_loss_ramp_shape bell
}

# ==============================================================================
# 阶段 3：三臂双 Best 全量 Test (N=1817) 统一评测 + 显著性检验 + 打包校验
# ==============================================================================
eval_and_package() {
    for ARM in esconv_case_baseline esconv_wo_all_proto esconv_case_epcl; do
        echo "======================================================================"
        echo "[Eval] 全量 ESConv Test 评测 (Greedy + Beam-5): ${ARM}"
        echo "======================================================================"
        ${PYTHON_BIN} research/esconv_generalization_20261122/eval_esconv_std.py \
            --save_dir "save/${ARM}" \
            --out_dir "save/${ARM}/eval_std" \
            --data_dir data \
            --gpu "${GPU_ID}" \
            --max_dec_step 50 \
            --lambda_epcl 0.03 2>&1 | tee -a "logs/${ARM}_eval.log"

        mkdir -p "${DOWNLOAD_ROOT}/${ARM}/eval_std"
        cp -f "logs/${ARM}.log" "logs/${ARM}_eval.log" "${DOWNLOAD_ROOT}/${ARM}/"
        cp -f "save/${ARM}"/CASE_*_composite_best "save/${ARM}"/CASE_*_ppl_best "${DOWNLOAD_ROOT}/${ARM}/"
        cp -f "save/${ARM}/eval_std"/* "${DOWNLOAD_ROOT}/${ARM}/eval_std/"
        sha256sum "save/${ARM}"/CASE_*_best > "${DOWNLOAD_ROOT}/${ARM}/checkpoint_sha256.txt"
    done

    ${PYTHON_BIN} research/esconv_generalization_20261122/eval_esconv_std.py \
        --compare \
        "baseline=save/esconv_case_baseline/eval_std" \
        "wo_all_proto=save/esconv_wo_all_proto/eval_std" \
        "case_epcl=save/esconv_case_epcl/eval_std" \
        --compare_out "${DOWNLOAD_ROOT}/esconv_3arms_summary.json"

    sha256sum main.py src/models/CASE/model.py src/utils/config.py src/utils/data/loader.py src/utils/decode/case.py \
        research/esconv_generalization_20261122/build_esconv_dataset.py \
        research/esconv_generalization_20261122/eval_esconv_std.py \
        research/esconv_generalization_20261122/run_esconv_3arms.sh \
        > "${DOWNLOAD_ROOT}/source_sha256.txt"

    echo "======================================================================"
    echo "[ALL DONE] ESConv 3 臂泛化实验全流程完成！汇总目录: ${DOWNLOAD_ROOT}"
    echo "======================================================================"
    ls -lh "${DOWNLOAD_ROOT}/"
}

case "${STAGE}" in
    prepare) prepare_data ;;
    train)   train_all_arms ;;
    eval)    eval_and_package ;;
    all)
        prepare_data
        train_all_arms
        eval_and_package
        ;;
    *)
        echo "未知阶段参数: ${STAGE}（支持 all | prepare | train | eval）" >&2
        exit 1
        ;;
esac
