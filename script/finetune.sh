#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="0,1,2,3,4,5"

TRAIN_ROOT="/data/lxl/NTIRE2026/train"
TRAIN_TXT="/data/lxl/NTIRE2026/train/train.txt"
VAL_TXT="/data/lxl/NTIRE2026/train/val.txt"
DATA_ROOT="/data/lxl/NTIRE2026"

OUT_DIR="/data/lxl/NTIRE2026/da3_nested_fp32_zero2_200epoch"
DS_CONFIG="ds_config.json"
BASE_MODEL="/data/lxl/NTIRE2026/models/DA3NESTED-GIANT-LARGE-1.1"

CAMERA="00"
BASELINE_UNIT="mm"

EPOCHS=300
BATCH=1
LR="5e-7"

# 建议用 14 的倍数
CROP_H=448
CROP_W=672

WARMUP_RATIO="0.05"
POLY_POWER="0.9"

MASTER_PORT="29601"

# no-K
USE_K_FLAG=""

AUX_METRIC_LOSS_WEIGHT="0.0"

W_HUBER="1.0"
HUBER_DELTA_CM="20.0"
W_ABSREL="0.25"
W_SILOG="0.10"
SILOG_LAMBDA="0.15"
W_GRAD="0.0"

mkdir -p /data/lxl/NTIRE2026/.triton_cache
export TRITON_CACHE_DIR=/data/lxl/NTIRE2026/.triton_cache

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${LOG_DIR}/finetune_$(date +%Y%m%d_%H%M%S).log"

CMD=(
  deepspeed
  --master_port "${MASTER_PORT}"
  --include "localhost:${GPU_LIST}"
  finetune.py
  --train_root "${TRAIN_ROOT}"
  --train_txt "${TRAIN_TXT}"
  --val_txt "${VAL_TXT}"
  --camera "${CAMERA}"
  --baseline_unit "${BASELINE_UNIT}"
  --out_dir "${OUT_DIR}"
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --lr "${LR}"
  --crop_h "${CROP_H}"
  --crop_w "${CROP_W}"
  --warmup_ratio "${WARMUP_RATIO}"
  --poly_power "${POLY_POWER}"
  --aux_metric_loss_weight "${AUX_METRIC_LOSS_WEIGHT}"
  --w_huber "${W_HUBER}"
  --huber_delta_cm "${HUBER_DELTA_CM}"
  --w_absrel "${W_ABSREL}"
  --w_silog "${W_SILOG}"
  --silog_lambda "${SILOG_LAMBDA}"
  --w_grad "${W_GRAD}"
  --ds_config "${DS_CONFIG}"
  --base_model "${BASE_MODEL}"
  --num_workers 0
)

if [[ -n "${USE_K_FLAG}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_FLAGS=(${USE_K_FLAG})
  CMD+=("${EXTRA_FLAGS[@]}")
fi

echo "[RUN] ${CMD[*]}"
echo "[PORT] ${MASTER_PORT}"
echo "[LOG] ${LOG_FILE}"

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "PID: ${PID}"
echo "LOG: ${LOG_FILE}"
echo "tail -f ${LOG_FILE}"

echo
echo "================ 推理命令（本次微调对应） ================"
echo "python infer_submit.py \\"
echo "  --data_root ${DATA_ROOT} \\"
echo "  --split val \\"
echo "  --camera camera_${CAMERA} \\"
echo "  --ckpt ${OUT_DIR}/best_16bit.pth \\"
echo "  --result_root ${DATA_ROOT}/results_val_camera_${CAMERA}_$(basename "${OUT_DIR}") \\"
echo "  --zip_path ${DATA_ROOT}/submission_val_camera_${CAMERA}_$(basename "${OUT_DIR}").zip"
echo "========================================================"
