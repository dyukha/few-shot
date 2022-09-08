cd ..
task=$1
task_name=$2
BS=2
LR=1000
num_labels=$3
GPU=$4
TAG=${task}_fewshot_similar_repeat

TYPE=prompt-demo

for i in "1 0.06251 1" "2 0.06251 2" "4 0.06251 4" "8 0.06251 8" "16 0.06251 16" \
         "2 0.1251 1" "4 0.1251 2" "8 0.1251 4" "16 0.1251 8" \
         "4 0.251 1" "8 0.251 2" "16 0.251 4" \
         "8 0.5001 1" "16 0.5001 2" \
         "16 1.0001 1"
do
  pair=( $i )
  K=${pair[0]}
  demo_filter_rate=${pair[1]}
  repeat=${pair[2]}

  dir=result/${TAG}_${K}_${demo_filter_rate}_${repeat}_similar_repeat
  mkdir -p $dir

  num_demos=$(echo $K \* $num_labels | bc)
  num_demos=$((num_demos>32 ? 32 : num_demos))

  array_id=0
  for SEED in 13 21 42 87 100
  do
    echo "Start fewshot with repeat $TAG $SEED $K $num_demos $demo_filter_rate on $GPU"

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
                            --demo_filter_type similar \
                            --demo_filter \
                            --demo_filter_model sbert-roberta-large \
                            --demo_filter_rate $demo_filter_rate \
                            --demo_repeat $repeat \
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
  python tools/ensemble.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'gpt3_in_context_num': $num_demos, 'few_shot_type': '$TYPE', 'demo_filter_rate': $demo_filter_rate}" --n_models 1 --save_logit_dir $dir
done


for i in "1 0.06251 1" "2 0.06251 2" "4 0.06251 4" "8 0.06251 8" "16 0.06251 16" \
         "2 0.1251 1" "4 0.1251 2" "8 0.1251 4" "16 0.1251 8" \
         "4 0.251 1" "8 0.251 2" "16 0.251 4" \
         "8 0.5001 1" "16 0.5001 2" \
         "16 1.0001 1"
do
  pair=( $i )
  K=${pair[0]}
  demo_filter_rate=${pair[1]}
  repeat=${pair[2]}

  num_demos=$(echo $K \* $num_labels | bc)
  num_demos=$((num_demos>32 ? 32 : num_demos))

  dir=result/${TAG}_${K}_${demo_filter_rate}_${repeat}_similar_repeat

  python tools/ensemble.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'gpt3_in_context_num': $num_demos, 'few_shot_type': '$TYPE', 'demo_filter_rate': $demo_filter_rate}" --n_models 1 --save_logit_dir $dir
done

#python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'few_shot_type': 'prompt-demo'}"
