cd ..
task=$1
task_name=$2
BS=2
LR=1000
num_labels=$3
GPU=$4
TAG=${task}_fewshot

TYPE=prompt-demo

for K in 1 2 4 8 16
do

  dir=result/${TAG}_${K}
  mkdir -p $dir

  num_demos=$(echo $K \* $num_labels | bc)
  num_demos=$((num_demos>32 ? 32 : num_demos))

  array_id=0
  for SEED in 13 21 42 87 100
  do
    echo "Start fewshot $TAG $SEED $K $num_demos on $GPU"

    TAG=$TAG \
    TYPE=$TYPE \
    TASK=$task \
    SEED=$SEED \
    BS=$BS \
    LR=$LR \
    MODEL=roberta-large \
    CUDA_VISIBLE_DEVICES=$GPU \
    K=16 \
    bash run_experiment_not_save.sh "--template_path auto_template/$task/16-$SEED.sort.txt \
                            --template_id 0 \
                            --no_train \
                            --num_sample 1 \
                            --gpt3_in_context_head \
                            --gpt3_in_context_num $num_demos
                            --truncate_head
                            --use_full_length
                            --save_logit \
                            --save_logit_dir $dir \
                            --model_id 0 \
                            --array_id $array_id"

  #  python tools/ensemble.py --condition "{'tag': '$TAG', 'task_name': '$task_name'}" --n_models 1 --save_logit_dir $dir
    array_id=$(expr $array_id + 1)
  done
  python tools/ensemble.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'gpt3_in_context_num': $num_demos, 'few_shot_type': '$TYPE'}" --n_models 1 --save_logit_dir $dir
done


for K in 1 2 4 8 16
do
  num_demos=$(echo $K \* $num_labels | bc)
  num_demos=$((num_demos>32 ? 32 : num_demos))

  dir=result/${TAG}_${K}

  python tools/ensemble.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'gpt3_in_context_num': $num_demos, 'few_shot_type': '$TYPE'}" --n_models 1 --save_logit_dir $dir
done

#python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'few_shot_type': 'prompt-demo'}"
