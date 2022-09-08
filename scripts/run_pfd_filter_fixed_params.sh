cd ..
task=$1
task_name=$2
SEED=$3
BS=$4
LR=$5
GPU=$6
TAG=${task}_prompt_demo_fitler_auto_fixed_params
echo "Start $TAG $SEED $BS $LR on $GPU"

TAG=$TAG \
TYPE=prompt-demo \
TASK=$task \
SEED=$SEED \
BS=$BS \
LR=$LR \
MODEL=roberta-large \
CUDA_VISIBLE_DEVICES=$GPU \
K=16 \
bash run_experiment_save.sh "--template_path auto_template/$task/16-$SEED.sort.txt --template_id 0 --demo_filter --demo_filter_model sbert-roberta-large"

python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task_name', 'few_shot_type': 'prompt-demo'}"
