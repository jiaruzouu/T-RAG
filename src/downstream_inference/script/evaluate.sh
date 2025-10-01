MODEL=(meta-llama/Llama-3.1-70B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct microsoft/Phi-3.5-mini-instruct)

DATASET=(tabfact) # tabfact, wtq, hybridqa, etc
TESTING_NUM=100

TOPK_LIST=(10 20 50)

for dataset in "${DATASET[@]}"; do
    for model in "${MODEL[@]}"; do
        for topk in "${TOPK_LIST[@]}"; do
            echo "Running with topk=$topk"
            python evaluation.py --dataset $DATASET --model $model --topk $topk --testing_num $TESTING_NUM
        done
    done
done