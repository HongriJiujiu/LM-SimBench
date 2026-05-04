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
MBS=2
GBS=128
SEQ_LEN=4096
TRAIN_ITERS=10
NUM_LAYERS=32
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TOKENIZER_PATH="your tokenizer directory path"

LONG_FACTOR="1.0800000429153442,1.1100000143051147,1.1399999856948853,1.340000033378601,1.5899999141693115,1.600000023841858,1.6200000047683716,2.620000123977661,3.2300000190734863,3.2300000190734863,4.789999961853027,7.400000095367432,7.700000286102295,9.09000015258789,12.199999809265137,17.670000076293945,24.46000099182129,28.57000160217285,30.420001983642578,30.840002059936523,32.590003967285156,32.93000411987305,42.320003509521484,44.96000289916992,50.340003967285156,50.45000457763672,57.55000305175781,57.93000411987305,58.21000289916992,60.1400032043457,62.61000442504883,62.62000274658203,62.71000289916992,63.1400032043457,63.1400032043457,63.77000427246094,63.93000411987305,63.96000289916992,63.970001220703125,64.02999877929688,64.06999969482422,64.08000183105469,64.12000274658203,64.41000366210938,64.4800033569336,64.51000213623047,64.52999877929688,64.83999633789062"
SHORT_FACTOR="1.0,1.0199999809265137,1.0299999713897705,1.0299999713897705,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0699999332427979,1.0999999046325684,1.1099998950958252,1.1599998474121094,1.1599998474121094,1.1699998378753662,1.2899998426437378,1.339999794960022,1.679999828338623,1.7899998426437378,1.8199998140335083,1.8499997854232788,1.8799997568130493,1.9099997282028198,1.9399996995925903,1.9899996519088745,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0799996852874756,2.0899996757507324,2.189999580383301,2.2199995517730713,2.5899994373321533,2.729999542236328,2.749999523162842,2.8399994373321533"

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
CP_LIST=(1 2 4)
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
   # PP > 1：启用虚拟流水线（Virtual Pipeline Parallelism）
   PP_ARGS="
   --pipeline-model-parallel-size ${PP} \
   --num-layers-per-virtual-pipeline-stage 1
   "
else
   # PP = 1：不启用任何流水线并行
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
    --tensor-model-parallel-size ${TP} \
    $PP_ARGS
    ${CP_ARGS}
    --sequence-parallel \
    --use-mcore-models \
    --rope-scaling-type longrope \
    --longrope-freqs-type outer \
    --rope-scaling-original-max-position-embeddings 4096 \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-fused-swiglu \
    --swiglu \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 3072 \
    --ffn-hidden-size 8192 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 131072 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    ${SP_ARGS}\
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --use-flash-attn \
    --rotary-base 10000 \
    --use-distributed-optimizer \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --padded-vocab-size 32064 \
    --untie-embeddings-and-output-weights \
    --seed 42 \
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
    --long-factor ${LONG_FACTOR} \
    --short-factor ${SHORT_FACTOR} \
    2>&1 | tee ${LOG_DIR}/pretrain.log
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "FAILED CONFIG: DP=${DP} TP=${TP} PP=${PP} CP=${CP} RC=${RC_NAME}"
    echo "Cleaning failed output..."
    rm -rf ${PROFILE_DIR}

else
    echo "SUCCESS CONFIG: DP=${DP} TP=${TP} PP=${PP} CP=${CP} RC=${RC_NAME}"
    echo "Deleting PROF* directories under ${PROFILE_DIR} ..."
    python "${PROJECT_DIR}/delete_PROF_000001.py" "${PROFILE_DIR}" || true

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
