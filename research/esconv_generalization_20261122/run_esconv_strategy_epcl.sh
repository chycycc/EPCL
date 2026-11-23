#!/bin/bash
# ==============================================================================
# ESConv 方案 B：原生 8 类轮次支持策略原型对齐实验（esconv_strategy_epcl）
#
# 核心假设验证：
#   - 对比原版 CASE-ESConv (esconv_case_baseline)：100% 相同 8 类策略监督（零额外情绪标注）、
#     相同 Session 防泄漏划分与超参配方，仅开启任务对齐的 CASE-EPCL 完整框架
#     (EPCL+MCP+MoP-DR+PCMG+PCAM+dec_str_loss+PCGrad+Valid-Neg+多粒度聚合)。
#   - 对比静态情绪原型版 (esconv_case_epcl)：验证将原型空间从“跨 14 轮恒定的 11 类负面会话情绪”
#     对齐至“轮次动态切换的 8 类支持策略”后，能否根治通用共情句塌缩并同步提升策略预测与生成质量。
#
# 用法：
#   bash research/esconv_generalization_20261122/run_esconv_strategy_epcl.sh [strategy_only|both|eval]
# ==============================================================================
set -e
set -o pipefail

MODE=${1:-"strategy_only"}
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "/root/miniconda3/bin/python" ]; then
        PYTHON_BIN="/root/miniconda3/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi
GPU_ID=${CUDA_VISIBLE_DEVICES:-"0"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs results/esconv
if [ -d "${HOME}/autodl-tmp" ]; then
    DOWNLOAD_ROOT="${HOME}/autodl-tmp/download/esconv_strategy_epcl"
else
    DOWNLOAD_ROOT="results/download/esconv_strategy_epcl"
fi
mkdir -p "${DOWNLOAD_ROOT}"

# Fail-Fast：确认预处理数据已就绪
if [ ! -s "data/ESConv/dataset_preproc.p" ]; then
    echo "[ERROR] 未检测到 data/ESConv/dataset_preproc.p，请先完成数据预处理。"
    exit 1
fi

# 与三臂基线 100% 严格一致的训练基础配方
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

train_strategy_epcl() {
    run_arm "esconv_strategy_epcl" \
        --esconv_enable_strategy_epcl \
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
        --use_pcam \
        --pcam_inner_ln \
        --pcam_heads 2 \
        --pcam_dropout 0.1 \
        --pcam_gate_bias -1.0 \
        --pcam_tau_attn 1.5 \
        --emo_branch_dropout 0.1 \
        --use_pcgrad
}

train_strategy_wo_all_proto() {
    run_arm "esconv_strategy_wo_all_proto" \
        --esconv_enable_strategy_epcl \
        --use_pcmg_head \
        --use_masked_react_pool \
        --use_fine_mim_valid_neg \
        --emo_branch_dropout 0.1 \
        --use_pcgrad \
        --disable_all_prototypes
}

eval_and_package() {
    local TARGET_ARMS=("esconv_strategy_epcl")
    if [ "${MODE}" = "both" ] || [ -d "save/esconv_strategy_wo_all_proto" ]; then
        if compgen -G "save/esconv_strategy_wo_all_proto/CASE_*_ppl_best" > /dev/null; then
            TARGET_ARMS+=("esconv_strategy_wo_all_proto")
        fi
    fi

    for ARM in "${TARGET_ARMS[@]}"; do
        echo "======================================================================"
        echo "[Eval] 全量 Test (N=1817, batch=1, Greedy + Beam-5): ${ARM}"
        echo "======================================================================"
        ${PYTHON_BIN} research/esconv_generalization_20261122/eval_esconv_std.py \
            --save_dir "save/${ARM}" \
            --out_dir "save/${ARM}/eval_std" \
            --data_dir data \
            --gpu "${GPU_ID}" \
            --max_dec_step 50 \
            --lambda_epcl 0.03 2>&1 | tee "logs/${ARM}_eval.log"

        mkdir -p "${DOWNLOAD_ROOT}/${ARM}/eval_std"
        cp -f "save/${ARM}/eval_std/"* "${DOWNLOAD_ROOT}/${ARM}/eval_std/"
        cp -f "save/${ARM}/config.txt" "${DOWNLOAD_ROOT}/${ARM}/" 2>/dev/null || true
        cp -f "logs/${ARM}.log" "logs/${ARM}_eval.log" "${DOWNLOAD_ROOT}/${ARM}/" 2>/dev/null || true
        cp -f "save/${ARM}/"CASE_*_best "${DOWNLOAD_ROOT}/${ARM}/"
        sha256sum "${DOWNLOAD_ROOT}/${ARM}/"CASE_*_best > "${DOWNLOAD_ROOT}/${ARM}/checkpoint_sha256.txt"
    done

    # 与已有三臂结果做完整 4 臂（或 5 臂）成对显著性汇总
    local CMP_ARGS=()
    if [ -s "save/esconv_case_baseline/eval_std/ppl_best_eval_std.json" ]; then
        CMP_ARGS+=("baseline=save/esconv_case_baseline/eval_std")
    fi
    if [ -s "save/esconv_wo_all_proto/eval_std/ppl_best_eval_std.json" ]; then
        CMP_ARGS+=("emo_wo_all_proto=save/esconv_wo_all_proto/eval_std")
    fi
    if [ -s "save/esconv_case_epcl/eval_std/ppl_best_eval_std.json" ]; then
        CMP_ARGS+=("emo_epcl=save/esconv_case_epcl/eval_std")
    fi
    if [ -s "save/esconv_strategy_wo_all_proto/eval_std/ppl_best_eval_std.json" ]; then
        CMP_ARGS+=("strategy_wo_all_proto=save/esconv_strategy_wo_all_proto/eval_std")
    fi
    CMP_ARGS+=("strategy_epcl=save/esconv_strategy_epcl/eval_std")

    ${PYTHON_BIN} research/esconv_generalization_20261122/eval_esconv_std.py \
        --compare "${CMP_ARGS[@]}" \
        --compare_out "${DOWNLOAD_ROOT}/esconv_strategy_compare_summary.json"

    sha256sum main.py src/models/CASE/model.py src/utils/config.py src/utils/data/loader.py src/utils/decode/case.py \
        research/esconv_generalization_20261122/eval_esconv_std.py \
        research/esconv_generalization_20261122/run_esconv_strategy_epcl.sh \
        > "${DOWNLOAD_ROOT}/source_sha256.txt"

    echo "======================================================================"
    echo "[ALL DONE] 策略原型对齐实验全部完成并已打包至: ${DOWNLOAD_ROOT}"
    ls -lh "${DOWNLOAD_ROOT}"
    echo "======================================================================"
}

case "${MODE}" in
    strategy_only)
        train_strategy_epcl
        eval_and_package
        ;;
    both)
        train_strategy_epcl
        train_strategy_wo_all_proto
        eval_and_package
        ;;
    eval)
        eval_and_package
        ;;
    *)
        echo "用法: $0 [strategy_only|both|eval]"
        exit 1
        ;;
esac
