cd ..
task=$1
task_name=$2
SEED=$3
BS=2
LR=1000
MODEL=$4
GPU=$5
TAG=${task}_prompt_demo_fitler_auto_fixed_params_infer

TYPE=prompt-demo

#for i in "1 0.06251" "2 0.1251" "4 0.251" "8 0.5001" "16 1.0001"

for i in "1 0.06251" "4 0.1251" "16 0.2501" "16 1.0001"
do
  pair=( $i )
  num_sample=${pair[0]}
  demo_filter_rate=${pair[1]}

  dir=result/${TAG}_${num_sample}_${demo_filter_rate}
  mkdir -p $dir


  echo "Start inference $TAG $seed $num_sample on $GPU"

  TAG=$TAG \
  TYPE=$TYPE \
  TASK=$task \
  SEED=$SEED \
  BS=$BS \
  LR=$LR \
  MODEL=$MODEL \
  CUDA_VISIBLE_DEVICES=$GPU \
  K=16 \
  bash run_experiment_not_save.sh "--template_path auto_template/$task/16-$SEED.sort.txt \
                          --template_id 0 \
                          --demo_filter \
                          --demo_filter_model sbert-roberta-large \
                          --num_sample $num_sample \
                          --demo_filter_rate $demo_filter_rate \
                          --no_train \
                          --save_logit \
                          --save_logit_dir $dir \
                          --model_id 0 \
                          --array_id 0"

#  python tools/ensemble.py --condition "{'tag': '$TAG', 'task_name': '$task_name'}" --n_models 1 --save_logit_dir $dir
  python tools/ensemble.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'num_sample': $num_sample, 'few_shot_type': '$TYPE', 'demo_filter_rate': $demo_filter_rate}" --n_models 1 --save_logit_dir $dir
done


for i in "1 0.06251" "2 0.1251" "4 0.251" "8 0.5001" "16 1.0001"
do
  pair=( $i )
  num_sample=${pair[0]}
  demo_filter_rate=${pair[1]}

  dir=result/${TAG}_${num_sample}_${demo_filter_rate}

  python tools/ensemble.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'num_sample': $num_sample, 'few_shot_type': '$TYPE', 'demo_filter_rate': $demo_filter_rate}" --n_models 1 --save_logit_dir $dir
done

#python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'few_shot_type': 'prompt-demo'}"
