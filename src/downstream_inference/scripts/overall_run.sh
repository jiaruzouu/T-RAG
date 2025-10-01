MODEL=(gpt-4o gpt-4o-mini) # Add more models if needed
DATASET=(sqa)
MODE=API # or "offline" for local open-source model inference, if offline, please set CUDA_VISIBLE_DEVICES in the loop
TESTIN_NUM=100
TOPK_LIST=(10 20 50)

for dataset in "${DATASET[@]}"; do
    for model in "${MODEL[@]}"; do
        for topk in "${TOPK_LIST[@]}"; do
            echo "Running with topk=$topk"
            python call_llm.py --dataset $dataset --topk $topk --mode $MODE --model $model --testing_num $TESTIN_NUM
            python evaluation.py --dataset $DATASET --model $model --topk $topk --testing_num $TESTIN_NUM
        done
    done
done