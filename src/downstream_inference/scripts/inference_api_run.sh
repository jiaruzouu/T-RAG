MODEL=(gpt-4o)
TEST_NUM=100
DATASET=(tabfact)
MODE=API
TOPK_LIST=(10 20 50)

for dataset in "${DATASET[@]}"; do
    for model in "${MODEL[@]}"; do
        for topk in "${TOPK_LIST[@]}"; do
            echo "Running with topk=$topk"
            python call_llm.py --dataset $dataset --topk $topk --mode $MODE --model $model --testing_num $TEST_NUM
        done
    done
done

