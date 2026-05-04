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
GBS=64
SEQ_LEN=4096
TRAIN_ITERS=10
NUM_LAYERS=32
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TOKENIZER_PATH="your tokenizer directory path"

LONG_FACTOR="1.0199999809265137,1.0299999713897705,1.0399999618530273,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.059999942779541,1.059999942779541,1.059999942779541,1.059999942779541,1.059999942779541,1.059999942779541,1.0999999046325684,1.1799999475479126,1.1799999475479126,1.3700000047683716,1.4899998903274536,2.109999895095825,2.8899998664855957,3.9499998092651367,4.299999713897705,6.429999828338623,8.09000015258789,10.690000534057617,12.050000190734863,18.229999542236328,18.84000015258789,19.899999618530273,21.420000076293945,26.200000762939453,34.28000259399414,34.590003967285156,38.730003356933594,40.22000503540039,42.54000473022461,44.000003814697266,47.590003967285156,54.750003814697266,56.19000244140625,57.44000244140625,57.4900016784668,61.20000076293945,61.540000915527344,61.75,61.779998779296875,62.06999969482422,63.11000061035156,63.43000030517578,63.560001373291016,63.71000289916992,63.92000198364258,63.94000244140625,63.94000244140625,63.96000289916992,63.980003356933594,64.0300064086914,64.0300064086914,64.0300064086914,64.04000854492188,64.10000610351562,64.19000244140625,64.20999908447266,64.75,64.95999908447266"
SHORT_FACTOR="1.0,1.0399999618530273,1.0399999618530273,1.0399999618530273,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.059999942779541,1.059999942779541,1.0699999332427979,1.0699999332427979,1.0699999332427979,1.0699999332427979,1.1399999856948853,1.159999966621399,1.159999966621399,1.159999966621399,1.159999966621399,1.1799999475479126,1.1999999284744263,1.3199999332427979,1.3399999141693115,1.3499999046325684,1.3999998569488525,1.4799998998641968,1.4999998807907104,1.589999794960022,1.6499998569488525,1.71999990940094,1.8999998569488525,1.9099998474121094,1.9099998474121094,1.9899998903274536,1.9999998807907104,1.9999998807907104,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.0999999046325684,2.319999933242798,2.419999837875366,2.5899999141693115,2.7899999618530273"

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

if [ "$PP" -gt 1 ]; then
   # PP > 1：启用虚拟流水线（Virtual Pipeline Parallelism）
   PP_ARGS="
   --pipeline-model-parallel-size ${PP} \
   --num-layers-per-virtual-pipeline-stage 2
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
    --expert-model-parallel-size ${EP} \
    --num-experts 16 \
    --moe-router-topk 2 \
    --moe-grouped-gemm \
    --moe-router-load-balancing-type sparsemixer_topk \
    --moe-input-jitter-eps 0.01 \
    --moe-token-dispatcher-type allgather \
    --moe-permutation-async-comm \
    --use-fused-moe-token-permute-and-unpermute
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.phi35_moe_spec layer_spec \
    --no-rope-fusion \
    --tensor-model-parallel-size ${TP} \
    $PP_ARGS
    ${CP_ARGS}
    --sequence-parallel \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 131072 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 6400 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --padded-vocab-size 32064 \
    --make-vocab-size-divisible-by 1 \
    --rotary-base 10000 \
    --train-iters ${TRAIN_ITERS} \
    --add-qkv-bias \
    --disable-bias-linear \
    --add-dense-bias \
    --add-output-layer-bias \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rope-scaling-type longrope \
    --longrope-freqs-type outer \
    --rope-scaling-original-max-position-embeddings 4096 \
    --long-mscale 1.243163121016122 \
    --short-mscale 1.243163121016122 \
    --long-factor ${LONG_FACTOR} \
    --short-factor ${SHORT_FACTOR} \
    --normalization LayerNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --seed 42 \
    --bf16 \
    ${RC_ARGS}
"

OPTIM_ARGS="
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-fused-swiglu \
    --use-fused-rotary-pos-emb \
    --attention-softmax-in-fp32 \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --no-load-optim \
    --no-load-rng
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    ${SP_ARGS}\
    --lr 1.25e-6 \
    --min-lr 1.25e-7 \
    --lr-decay-style cosine \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --trust-remote-code
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
    $MOE_ARGS \
    $GPT_ARGS \
    $OPTIM_ARGS \
    $TRAIN_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
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
