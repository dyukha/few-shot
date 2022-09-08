cd ..
task=$1
task_name=$2
SEED=$3
BS=2
LR=1000
MODEL=$4
GPU=$5
TAG=${task}_prompt_demo_fitler_auto_fixed_params_infer

for num_sample in 1 2 4 8 16
do
  dir=result/${TAG}_${num_sample}_0.5
  mkdir -p $dir

  echo "Start inference $TAG $seed $num_sample on $GPU"
  TAG=$TAG \
  TYPE=prompt-demo \
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
                          --no_train \
                          --save_logit \
                          --save_logit_dir $dir \
                          --model_id 0 \
                          --array_id 0"

#  python tools/ensemble.py --condition "{'tag': '$TAG', 'task_name': '$task_name'}" --n_models 1 --save_logit_dir $dir
  python tools/ensemble.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'num_sample': $num_sample, 'few_shot_type': 'prompt-demo'}" --n_models 1 --save_logit_dir $dir
done


for num_sample in 1 2 4 8 16
do
  dir=result/${TAG}_${num_sample}_0.5
  python tools/ensemble.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'num_sample': $num_sample, 'few_shot_type': 'prompt-demo'}" --n_models 1 --save_logit_dir $dir
done

#python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'few_shot_type': 'prompt-demo'}"
