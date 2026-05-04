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
    local timeout_sec=6000
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
NNODES=2
MBS=1
GBS=256
SEQ_LEN=4096
TRAIN_ITERS=10
NUM_LAYERS=48
CP_TYPE='ulysses_cp_algo'
ROUTER_BALANCING_TYPE='aux_loss'
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

TP_LIST=(4 8 16)
PP_LIST=(2 4 8 16)
EP_LIST=(1 2 4 8 16)
CP_LIST=(1 2 4 8 16)
RC_LIST=(0 1)
MAX_DP=2

for TP in "${TP_LIST[@]}"
do
for PP in "${PP_LIST[@]}"
do
for EP in "${EP_LIST[@]}"
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

TOTAL=$((TP*PP*CP*EP))

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

RUN_NAME="dp${DP}-tp${TP}-ep${EP}-pp${PP}-cp${CP}-${RC_NAME}"

MODEL_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
LOG_DIR="${OUTPUT_ROOT}/log/${MODEL_NAME}/${RUN_NAME}"
PROFILE_DIR="${OUTPUT_ROOT}/profiler/${MODEL_NAME}/${RUN_NAME}"
BARRIER_DIR="${OUTPUT_ROOT}/barrier/${MODEL_NAME}/${RUN_NAME}"
mkdir -p ${LOG_DIR}
mkdir -p ${PROFILE_DIR}
rm -rf ${BARRIER_DIR}

echo "===================================="
echo "Running configuration:"
echo "DP=${DP} TP=${TP} EP=${EP} PP=${PP} CP=${CP} RC=${RC_NAME}"
echo "===================================="

MASTER_PORT=$((20000 + TP*1000 + PP*100 + CP*10 + EP*3 + RC))
DISTRIBUTED_ARGS="
--nproc_per_node $NPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-intermediate-size 768 \
    --moe-grouped-gemm \
    --use-fused-moe-token-permute-and-unpermute \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-aux-loss-coeff 0.001 \
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
    --gemm-gradient-accumulation-fusion \
    --recompute-method uniform \
    --recompute-granularity full \
    --recompute-num-layers 1 \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    ${SP_ARGS}\
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LEN} \
    --no-shared-storage
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    $PP_ARGS
    ${CP_ARGS}
    --expert-model-parallel-size ${EP} \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LEN} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 4 \
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
    --no-load-optim \
    --no-load-rng
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
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    $PROFILE \
    --distributed-backend nccl \
    2>&1 | tee ${LOG_DIR}/pretrain.log
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "FAILED CONFIG: DP=${DP} TP=${TP} EP=${EP} PP=${PP} CP=${CP} RC=${RC_NAME}"
    echo "Cleaning failed output..."
    rm -rf ${PROFILE_DIR}

else
    echo "SUCCESS CONFIG: DP=${DP} TP=${TP} EP=${EP} PP=${PP} CP=${CP} RC=${RC_NAME}"

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
done
