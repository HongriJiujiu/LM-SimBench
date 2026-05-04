#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Install CANN, MindSpeed, MindSpeed-LLM and related dependencies before running.
# Please fill these path configurations for your environment.
PROJECT_DIR="your MindSpeed-LLM project directory path"
OUTPUT_ROOT="your output directory path"

cleanup_dist() {
    echo "[CLEANUP] killing old distributed processes..."
    pkill -f "torchrun" || true
    pkill -f "train_npu.py" || true
    sleep 5

    echo "[CLEANUP] port usage:"
    ss -lntp | grep ":${MASTER_PORT}" || true

    echo "[CLEANUP] remaining related processes:"
    ps -ef | grep -E "torchrun|train_npu.py|python" | grep -v grep || true
}

barrier_sync() {
    local timeout_sec=3000
    local waited=0

    mkdir -p "${BARRIER_DIR}"
    touch "${BARRIER_DIR}/node_${NODE_RANK}.done"

    echo "[BARRIER] node ${NODE_RANK} reached barrier: ${RUN_NAME}"

    while true; do
        count=$(find "${BARRIER_DIR}" -maxdepth 1 -name 'node_*.done' | wc -l)
        if [ "${count}" -ge "${NNODES}" ]; then
            break
        fi

        sleep 2
        waited=$((waited + 2))

        if [ "${waited}" -ge "${timeout_sec}" ]; then
            echo "[BARRIER] timeout waiting for all nodes at barrier: ${RUN_NAME}"
            return 1
        fi
    done

    echo "[BARRIER] all ${NNODES} nodes reached barrier: ${RUN_NAME}"
    sleep 3
}

# =========================
# 基础配置
# =========================

NPUS_PER_NODE=8
NNODES=1
MBS=1
GBS=64
SEQ_LEN=8192
TRAIN_ITERS=10
NUM_LAYERS=24
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TOKENIZER_PATH="your tokenizer directory path"

# =========================
# 分布式环境
# =========================

# Distributed environment. Override these variables for multinode runs.
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}

# =========================
# 枚举 TP / PP
# =========================

TP_LIST=(1 2 4 8)
PP_LIST=(1 2 4 8)
CP_LIST=(1 2 4 8)
RC_LIST=(0 1)
MAX_DP=8

for TP in "${TP_LIST[@]}"
do
for PP in "${PP_LIST[@]}"
do
for CP in "${CP_LIST[@]}"
do
CP_TYPE="ulysses_cp_algo"
if [ "$CP" -gt 1 ]; then
    CP_ARGS="
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE}
    "
else
    CP_ARGS=""
fi

TOTAL=$((TP*PP*CP))

if [ $((WORLD_SIZE % TOTAL)) -ne 0 ]; then
    continue
fi
DP=$((WORLD_SIZE / TOTAL))
if [ "$DP" -gt "$MAX_DP" ]; then
    continue
fi
if [ $((NUM_LAYERS % PP)) -ne 0 ]; then
    continue
fi

# 构建 num-layer-list
LAYER_PER_STAGE=$((NUM_LAYERS / PP))
LAYER_LIST=$(yes ${LAYER_PER_STAGE} | head -n ${PP} | paste -sd ",")
if [ "$PP" -gt 1 ]; then
    PP_ARGS="
    --pipeline-model-parallel-size ${PP} \
    --num-layer-list ${LAYER_LIST}
    "
else
    PP_ARGS=""
fi

for RC in "${RC_LIST[@]}"
do

if [ "$RC" -eq 1 ]; then
    RC_NAME="RC"
    RC_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers ${NUM_LAYERS}
    "
else
    RC_NAME="noRC"
    RC_ARGS=""
fi

if [ "$TP" -gt 1 ]; then
    SP_ARGS="--sequence-parallel"
else
    SP_ARGS=""
fi

RUN_NAME="dp${DP}-tp${TP}-pp${PP}-cp${CP}-${RC_NAME}"

MODEL_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
LOG_DIR="${OUTPUT_ROOT}/log/${MODEL_NAME}/${RUN_NAME}"
PROFILE_DIR="${OUTPUT_ROOT}/profiler/${MODEL_NAME}/${RUN_NAME}"
BARRIER_DIR="${OUTPUT_ROOT}/barrier/${MODEL_NAME}/${RUN_NAME}"
mkdir -p ${LOG_DIR}
mkdir -p ${PROFILE_DIR}
rm -rf ${BARRIER_DIR}

echo "===================================="
echo "Running configuration:"
echo "DP=${DP} TP=${TP} PP=${PP} CP=${CP} RC=${RC_NAME}"
echo "===================================="

MASTER_PORT=$((20000 + TP*1000 + PP*100 + CP*10 + RC))
DISTRIBUTED_ARGS="
--nproc_per_node $NPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    $PP_ARGS
    ${CP_ARGS}
    --sequence-parallel \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 2048 \
    --ffn-hidden-size 5504 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 32768 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    ${SP_ARGS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --lr 1.25e-6 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --add-qkv-bias \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --bf16 \
    ${RC_ARGS}
"

DATA_ARGS="
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
"

PROFILE="
--profile \
--profile-level level2 \
--profile-step-start  5 \
--profile-step-end 6 \
--profile-with-cpu \
--profile-with-stack \
--profile-with-memory \
--profile-record-shapes \
--profile-save-path ${PROFILE_DIR}
"

torchrun $DISTRIBUTED_ARGS "${PROJECT_DIR}/train_npu.py" \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $PROFILE \
    --distributed-backend nccl \
    2>&1 | tee ${LOG_DIR}/pretrain.log
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "FAILED CONFIG: DP=${DP} TP=${TP} PP=${PP} CP=${CP} RC=${RC_NAME}"
    echo "Cleaning failed output..."
    rm -rf ${PROFILE_DIR}

else
    echo "SUCCESS CONFIG: DP=${DP} TP=${TP} PP=${PP} CP=${CP} RC=${RC_NAME}"

fi

# cleanup_dist
if ! barrier_sync; then
    echo "[FATAL] barrier failed for ${RUN_NAME}, start cleanup..."
    cleanup_dist
    exit 1
fi
echo "Deleting PROF* directories under ${PROFILE_DIR} ..."
python "${PROJECT_DIR}/delete_PROF_000001.py" "${PROFILE_DIR}" || true
rm -f "${BARRIER_DIR}/node_${NODE_RANK}.done"
sleep 5

done
done
done
done
